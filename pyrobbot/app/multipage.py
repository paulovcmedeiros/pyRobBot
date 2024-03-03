"""Code for the creation streamlit apps with dynamically created pages."""

import contextlib
import datetime
import hashlib
import os
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict, deque
from json.decoder import JSONDecodeError

import streamlit as st
import streamlit_webrtc
from loguru import logger
from pydantic import ValidationError
from pydub import AudioSegment
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode

from pyrobbot import GeneralDefinitions
from pyrobbot.chat_configs import VoiceChatConfigs
from pyrobbot.general_utils import trim_beginning
from pyrobbot.openai_utils import OpenAiClientWrapper

from .app_page_templates import AppPage, ChatBotPage, _RecoveredChat
from .app_utils import (
    WebAppChat,
    filter_page_info_from_queue,
    get_avatar_images,
    get_ice_servers,
)

incoming_frame_queue = queue.Queue()
possible_speech_chunks_queue = queue.Queue()
audio_playing_chunks_queue = queue.Queue()
continuous_user_prompt_queue = queue.Queue()
text_prompt_queue = queue.Queue()
reply_ongoing = threading.Event()


@st.cache_resource(show_spinner="Initialising listening engine...")
def listen():  # noqa: PLR0912, PLR0915
    """Listen for speech from the browser."""
    # This deque will be employed to keep a moving window of audio chunks to monitor
    # voice activity. The length of the deque is calculated such that the concatenated
    # audio chunks will produce an audio at most inactivity_timeout_seconds long
    #
    # Mind that none of Streamlit's APIs are safe to call from any thread other than
    # the main one. See, e.g., <https://discuss.streamlit.io/t/
    # changing-session-state-not-reflecting-in-active-python-thread/37683
    logger.debug("Listener thread started")

    all_users_audio_chunks_moving_windows = {}
    user_has_been_speaking = defaultdict(lambda: False)
    all_users_moving_window_speech_likelihood = defaultdict(lambda: 0.0)

    while True:
        try:
            logger.trace("Waiting for audio frame from the stream...")
            received_audio_frame_info = incoming_frame_queue.get()
            received_audio_frame = received_audio_frame_info["frame"]

            app_page = received_audio_frame_info["page"]
            chat_obj = app_page.chat_obj
            try:
                audio_chunks_moving_window = all_users_audio_chunks_moving_windows[
                    app_page.page_id
                ]
            except KeyError:
                audio_chunks_moving_window = deque(
                    maxlen=int(
                        (1000.0 * chat_obj.inactivity_timeout_seconds)
                        / chat_obj.frame_duration
                    )
                )
                all_users_audio_chunks_moving_windows[app_page.page_id] = (
                    audio_chunks_moving_window
                )

            logger.trace(
                "Received audio frame from the stream on page '{}', chat {}",
                app_page.title,
                chat_obj.id,
            )

            moving_window_speech_likelihood = all_users_moving_window_speech_likelihood[
                app_page.page_id
            ]

            if received_audio_frame.sample_rate != chat_obj.sample_rate:
                raise ValueError(
                    f"audio_frame.sample_rate = {received_audio_frame.sample_rate} "
                    f"!= chat_obj.sample_rate = {chat_obj.sample_rate}"
                )

            # Convert the received audio frame to an AudioSegment object
            raw_samples = received_audio_frame.to_ndarray()
            audio_chunk = AudioSegment(
                data=raw_samples.tobytes(),
                sample_width=received_audio_frame.format.bytes,
                frame_rate=received_audio_frame.sample_rate,
                channels=len(received_audio_frame.layout.channels),
            )
            if audio_chunk.duration_seconds != chat_obj.frame_duration / 1000:
                raise ValueError(
                    f"sound_chunk.duration_seconds = {audio_chunk.duration_seconds} "
                    "!= chat_obj.frame_duration / 1000 = "
                    f"{chat_obj.frame_duration / 1000}"
                )

            # Resample the AudioSegment to be compatible with the VAD engine
            audio_chunk = audio_chunk.set_frame_rate(chat_obj.sample_rate).set_channels(1)

            # Now do the VAD
            # Check if the current sound chunk is likely to be speech
            vad_thinks_this_chunk_is_speech = chat_obj.vad.is_speech(
                audio_chunk.raw_data, chat_obj.sample_rate
            )

            # Monitor voice activity within moving window of length
            # inactivity_timeout_seconds
            audio_chunks_moving_window.append(
                {"audio": audio_chunk, "is_speech": vad_thinks_this_chunk_is_speech}
            )
            all_users_audio_chunks_moving_windows[app_page.page_id] = (
                audio_chunks_moving_window
            )

            moving_window_length = len(audio_chunks_moving_window)
            if moving_window_length == audio_chunks_moving_window.maxlen:
                voice_activity = (
                    chunk["is_speech"] for chunk in audio_chunks_moving_window
                )
                moving_window_speech_likelihood = (
                    sum(voice_activity) / moving_window_length
                )
                all_users_moving_window_speech_likelihood[app_page.page_id] = (
                    moving_window_speech_likelihood
                )

            user_speaking_now = (
                moving_window_speech_likelihood >= chat_obj.speech_likelihood_threshold
            )
            logger.trace("User speaking: {}", user_speaking_now)
            if user_has_been_speaking[app_page.page_id]:
                speech_chunk_info = {"audio": audio_chunk, "page": app_page}
                possible_speech_chunks_queue.put(speech_chunk_info)
                if not user_speaking_now:
                    user_has_been_speaking[app_page.page_id] = False
                    speech_chunk_info = {"audio": None, "page": app_page}
                    possible_speech_chunks_queue.put(speech_chunk_info)
                    logger.info("No more voice activity detected. Signal end of speech.")
                    continue
            elif user_speaking_now:
                logger.info("Voice activity detected")
                user_has_been_speaking[app_page.page_id] = True
                for past_audio_chunk in audio_chunks_moving_window:
                    speech_chunk_info = {
                        "audio": past_audio_chunk["audio"],
                        "page": app_page,
                    }
                    possible_speech_chunks_queue.put(speech_chunk_info)
        except Exception as error:  # noqa: BLE001
            logger.opt(exception=True).debug(error)
            logger.error(error)
        finally:
            incoming_frame_queue.task_done()


@st.cache_resource(show_spinner="Initialising listening engine...")
def handle_continuous_user_prompt():
    """Play audio."""
    logger.debug("Continuous user audio prompt handling thread started")
    while True:
        try:
            logger.trace("Waiting for new speech chunk...")
            new_audio_chunk_info = possible_speech_chunks_queue.get()
            new_audio_chunk = new_audio_chunk_info["audio"]
            app_page = new_audio_chunk_info["page"]
            chat_obj = app_page.chat_obj

            logger.trace("Processing new speech chunk for page '{}'", app_page.title)
            if new_audio_chunk is None:
                # User has stopped speaking. Concatenate all audios from
                # play_audio_queue and send the result to be played

                logger.debug(
                    "Gathering {} frames received to send as user input for page '{}'",
                    audio_playing_chunks_queue.qsize(),
                    app_page.title,
                )
                concatenated_audio = AudioSegment.empty()
                with audio_playing_chunks_queue.mutex:
                    this_page_audio_chunks = filter_page_info_from_queue(
                        app_page=app_page, the_queue=audio_playing_chunks_queue
                    )
                    while this_page_audio_chunks.queue:
                        audio_chunk_info = this_page_audio_chunks.queue.popleft()
                        concatenated_audio += audio_chunk_info["audio"]

                logger.debug(
                    "Done gathering frames ({}s) for page '{}'. Trimming...",
                    concatenated_audio.duration_seconds,
                    app_page.title,
                )
                concatenated_audio = trim_beginning(concatenated_audio)
                if (
                    concatenated_audio.duration_seconds
                    >= chat_obj.min_speech_duration_seconds
                ):
                    logger.debug(
                        'Page "{}": Make sure the queue has only the latest audio...',
                        app_page.title,
                    )
                    with continuous_user_prompt_queue.mutex:
                        filter_page_info_from_queue(
                            app_page=app_page, the_queue=continuous_user_prompt_queue
                        )

                    new_info_for_stt = {"page": app_page, "audio": concatenated_audio}
                    continuous_user_prompt_queue.put(new_info_for_stt)
                    logger.debug("Audio input for page '{}' sent for STT", app_page.title)
                else:
                    logger.debug(
                        'Page "{}": audio input too short ({} < {} sec). Discarding.',
                        app_page.title,
                        concatenated_audio.duration_seconds,
                        chat_obj.min_speech_duration_seconds,
                    )
            else:
                new_audio_chunk_info = {"page": app_page, "audio": new_audio_chunk}
                audio_playing_chunks_queue.put(new_audio_chunk_info)

            possible_speech_chunks_queue.task_done()
        except Exception as error:  # noqa: BLE001, PERF203
            logger.opt(exception=True).debug(error)
            logger.error(error)


@st.cache_resource(show_spinner="Initialising listening engine...")
def handle_stt():
    """Handle speech to text."""
    logger.debug("Speech to text handling thread started")

    while True:
        try:
            info_for_stt = continuous_user_prompt_queue.get()
            audio = info_for_stt["audio"]
            chat_obj = info_for_stt["page"].chat_obj
            if audio.duration_seconds >= chat_obj.min_speech_duration_seconds:
                recorded_prompt_as_txt = chat_obj.stt(audio).text
                if recorded_prompt_as_txt:
                    logger.debug(
                        "Audio from page '{}' transcribed  '{}'. Input ready to fetch.",
                        info_for_stt["page"].title,
                        recorded_prompt_as_txt,
                    )
                    text_prompt_queue.put(
                        {"page": info_for_stt["page"], "text": recorded_prompt_as_txt}
                    )
        except Exception as error:  # noqa: BLE001, PERF203
            logger.error(error)


listen_thread = threading.Thread(name="listener_thread", target=listen, daemon=True)
continuous_user_prompt_thread = threading.Thread(
    name="continuous_user_prompt_thread",
    target=handle_continuous_user_prompt,
    daemon=True,
)
handle_stt_thread = threading.Thread(
    name="stt_handling_thread", target=handle_stt, daemon=True
)


class AbstractMultipageApp(ABC):
    """Framework for creating streamlite multipage apps.

    Adapted from:
    <https://towardsdatascience.com/
     creating-multipage-applications-using-streamlit-efficiently-b58a58134030>.

    """

    def __init__(self, **kwargs) -> None:
        """Initialise streamlit page configs."""
        st.set_page_config(**kwargs)

        self.listen_thread = listen_thread
        self.continuous_user_prompt_thread = continuous_user_prompt_thread
        self.handle_stt_thread = handle_stt_thread
        if (
            st.session_state.get("toggle_continuous_voice_input")
            and not self.continuous_audio_input_engine_is_running
        ):
            for thread in [
                listen_thread,
                continuous_user_prompt_thread,
                handle_stt_thread,
            ]:
                # See <https://github.com/streamlit/streamlit/issues/
                #      1326#issuecomment-1597918085>
                add_script_run_ctx(thread)
                thread.start()

        self.incoming_frame_queue = incoming_frame_queue
        self.possible_speech_chunks_queue = possible_speech_chunks_queue
        self.audio_playing_chunks_queue = audio_playing_chunks_queue
        self.continuous_user_prompt_queue = continuous_user_prompt_queue
        self.text_prompt_queue = text_prompt_queue
        self.reply_ongoing = reply_ongoing

    @property
    def ice_servers(self):
        """Return the ICE servers for WebRTC."""
        return get_ice_servers()

    @property
    def continuous_audio_input_engine_is_running(self):
        """Return whether the continuous audio input engine is running."""
        return (
            self.listen_thread.is_alive()
            and self.continuous_user_prompt_thread.is_alive()
            and self.handle_stt_thread.is_alive()
        )

    def render_continuous_audio_input_widget(self):
        """Render the continuous audio input widget using webrtc_streamer."""
        try:
            selected_page = self.selected_page
        except StopIteration:
            selected_page = None

        def audio_frame_callback(frame):
            try:
                logger.trace("Received raw audio frame from the stream")

                if selected_page is None:
                    logger.trace("No page selected. Discardig audio chunk")
                    return frame

                if self.reply_ongoing.is_set():
                    logger.trace("Reply is ongoing. Discardig audio chunk")
                    return frame

                if not self.continuous_user_prompt_queue.empty():
                    logger.trace(
                        "Audio input queue not empty {} items). Discardig chunk",
                        self.continuous_user_prompt_queue.qsize(),
                    )
                    return frame
            except Exception as error:  # noqa: BLE001
                logger.opt(exception=True).debug(error)
                logger.error(error)
            else:
                frame_info = {"frame": frame, "page": selected_page}
                self.incoming_frame_queue.put(frame_info)
                logger.trace("Raw audio frame sent to the processing queue")

            return frame

        add_script_run_ctx(audio_frame_callback)

        logger.debug("Initialising input audio stream...")
        hide_webrtc_streamer_button = """
            <style>
            .element-container:has(
                iframe[title="streamlit_webrtc.component.webrtc_streamer"]
            ) {
              display: none;
              overflow: hidden;
              max-height: 0;
            }
            </style>
            """
        st.markdown(hide_webrtc_streamer_button, unsafe_allow_html=True)

        try:
            self.stream_audio_context = streamlit_webrtc.component.webrtc_streamer(
                key="sendonly-audio",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration={"iceServers": self.ice_servers},
                media_stream_constraints={"audio": True, "video": False},
                desired_playing_state=True,
                audio_frame_callback=audio_frame_callback,
            )
        except TypeError:
            logger.opt(exception=True).error("Failed to initialise audio stream")
            logger.error("Failed to initialise audio stream")
            self.stream_audio_context = None
        else:
            logger.debug("Audio stream initialised. Waiting for it to start...")
            while not self.stream_audio_context.state.playing:
                time.sleep(1)
            logger.debug("Audio stream started")

        return self.stream_audio_context

    @property
    def n_created_pages(self):
        """Return the number of pages created by the app, including deleted ones."""
        return self.state.get("n_created_pages", 0)

    @n_created_pages.setter
    def n_created_pages(self, value):
        self.state["n_created_pages"] = value

    @property
    def pages(self) -> dict[AppPage]:
        """Return the pages of the app."""
        if "available_pages" not in self.state:
            self.state["available_pages"] = {}
        return self.state["available_pages"]

    def add_page(self, page: AppPage, selected: bool = True, **page_obj_kwargs):
        """Add a page to the app."""
        if page is None:
            page = AppPage(parent=self, **page_obj_kwargs)

        self.pages[page.page_id] = page
        self.n_created_pages += 1
        if selected:
            self.selected_page = page

    def _remove_page(self, page: AppPage):
        """Remove a page from the app."""
        self.pages[page.page_id].chat_obj.clear_cache()
        del self.pages[page.page_id]
        try:
            self.selected_page = next(iter(self.pages.values()))
        except StopIteration:
            self.add_page()

    def remove_page(self, page: AppPage):
        """Remove a page from the app after confirmation."""
        st.error("Are you sure you want to delete this chat?")
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.button("No, take me back", use_container_width=True)
        with col2:
            st.button(
                "Yes, delete chat",
                on_click=self._remove_page,
                kwargs={"page": page},
                use_container_width=True,
            )

    @property
    def selected_page(self) -> ChatBotPage:
        """Return the selected page."""
        if "selected_page" not in self.state:
            self.selected_page = next(iter(self.pages.values()), None)
        return self.state["selected_page"]

    @selected_page.setter
    def selected_page(self, page: ChatBotPage):
        self.state["selected_page"] = page
        st.session_state["currently_active_page"] = page

    def render(self, **kwargs):
        """Render the multipage app with focus on the selected page."""
        self.handle_ui_page_selection(**kwargs)
        self.selected_page.render()
        self.state["last_rendered_page"] = self.selected_page.page_id

    @abstractproperty
    def state(self):
        """Return the state of the app, for persistence of data."""

    @abstractmethod
    def handle_ui_page_selection(self, **kwargs):
        """Control page selection in the UI sidebar."""


class MultipageChatbotApp(AbstractMultipageApp):
    """A Streamlit multipage app specifically for chatbot interactions.

    Inherits from AbstractMultipageApp and adds chatbot-specific functionalities.

    """

    @property
    def current_user_id(self):
        """Return the user id."""
        return hashlib.sha256(self.openai_api_key.encode("utf-8")).hexdigest()

    @property
    def current_user_st_state_id(self):
        """Return the user id for streamlit state."""
        return f"app_state_{self.current_user_id}"

    @property
    def state(self):
        """Return the state of the app, for persistence of data."""
        user_st_state_key = self.current_user_st_state_id
        if user_st_state_key not in st.session_state:
            st.session_state[user_st_state_key] = {}
        return st.session_state[user_st_state_key]

    @property
    def openai_client(self) -> OpenAiClientWrapper:
        """Return the OpenAI client."""
        if "openai_client" not in self.state:
            logger.debug("Creating OpenAI client for multipage app")
            self.state["openai_client"] = OpenAiClientWrapper(
                api_key=self.openai_api_key, private_mode=self.chat_configs.private_mode
            )
            logger.debug("OpenAI client created for multipage app")
        return self.state["openai_client"]

    @property
    def chat_configs(self) -> VoiceChatConfigs:
        """Return the configs used for the page's chat object."""
        if "chat_configs" not in self.state:
            try:
                chat_options_file_path = sys.argv[-1]
                self.state["chat_configs"] = VoiceChatConfigs.from_file(
                    chat_options_file_path
                )
            except (FileNotFoundError, JSONDecodeError):
                logger.warning("Could not retrieve cli args. Using default chat options.")
                self.state["chat_configs"] = VoiceChatConfigs()
        return self.state["chat_configs"]

    def create_api_key_element(self):
        """Create an input element for the OpenAI API key."""
        self.openai_api_key = st.text_input(
            label="OpenAI API Key (required)",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="Enter your OpenAI API key",
            key="openai_api_key",
            type="password",
            help="[OpenAI API auth key](https://platform.openai.com/account/api-keys). "
            + "Chats created with this key won't be visible to people using other keys.",
        )
        if not self.openai_api_key:
            st.write(":red[You need a valid key to use the chat]")

    def add_page(
        self, page: ChatBotPage = None, selected: bool = True, **page_obj_kwargs
    ):
        """Adds a new ChatBotPage to the app.

        If no page is specified, a new instance of ChatBotPage is created and added.

        Args:
            page: The ChatBotPage to be added. If None, a new page is created.
            selected: Whether the added page should be selected immediately.
            **page_obj_kwargs: Additional keyword arguments for ChatBotPage creation.

        Returns:
            The result of the superclass's add_page method.

        """
        if page is None:
            logger.debug("Resquest to add page without passing a page. Creating defaut.")
            page = ChatBotPage(parent=self, **page_obj_kwargs)
        else:
            logger.debug("Resquest to a specific page. Adding it.")
        return super().add_page(page=page, selected=selected)

    def get_widget_previous_value(self, widget_key, default=None):
        """Get the previous value of a widget, if any."""
        if "widget_previous_value" not in self.selected_page.state:
            self.selected_page.state["widget_previous_value"] = {}
        return self.selected_page.state["widget_previous_value"].get(widget_key, default)

    def save_widget_previous_values(self, element_key):
        """Save a widget's 'previous value`, to be read by `get_widget_previous_value`."""
        if "widget_previous_value" not in self.selected_page.state:
            self.selected_page.state["widget_previous_value"] = {}
        self.selected_page.state["widget_previous_value"][element_key] = (
            st.session_state.get(element_key)
        )

    def handle_ui_page_selection(self):
        """Control page selection and removal in the UI sidebar."""
        _set_button_style()
        self._build_sidebar_tabs()

        with self.sidebar_tabs["settings"]:
            caption = f"\u2699\uFE0F {self.selected_page.title}"
            st.caption(caption)
            current_chat_configs = self.selected_page.chat_obj.configs

            # Present the user with the model and instructions fields first
            field_names = ["model", "ai_instructions", "context_model"]
            field_names += list(VoiceChatConfigs.model_fields)
            field_names = list(dict.fromkeys(field_names))
            model_fields = {k: VoiceChatConfigs.model_fields[k] for k in field_names}

            updates_to_chat_configs = self._handle_chat_configs_value_selection(
                current_chat_configs, model_fields
            )
            if updates_to_chat_configs:
                current_chat_configs = self.selected_page.chat_obj.configs.copy()
                new_configs = current_chat_configs.model_dump()
                new_configs.update(updates_to_chat_configs)
                new_configs = self.selected_page.chat_obj.configs.model_validate(
                    new_configs
                )
                if new_configs != current_chat_configs:
                    logger.debug(
                        "Chat configs for page <{}> changed. Update page chat <{}>",
                        self.selected_page.sidebar_title,
                        self.selected_page.chat_obj.id,
                    )
                    self.selected_page.chat_obj = WebAppChat.from_dict(new_configs)

    def render(self, **kwargs):
        """Renders the multipage chatbot app in the  UI according to the selected page."""
        with st.sidebar:
            _left_col, centre_col, _right_col = st.columns([0.33, 0.34, 0.33])
            with centre_col:
                st.title(GeneralDefinitions.APP_NAME)
                with contextlib.suppress(AttributeError, ValueError, OSError):
                    # st image raises some exceptions occasionally
                    avatars = get_avatar_images()
                    st.image(avatars["assistant"], use_column_width=True)
            st.subheader(
                GeneralDefinitions.PACKAGE_DESCRIPTION,
                divider="rainbow",
                help="https://github.com/paulovcmedeiros/pyRobBot",
            )

            self.create_api_key_element()

            # Create a sidebar with tabs for chats and settings
            tab1, tab2 = st.tabs(["Chats", "Settings for Current Chat"])

            with tab1:
                tab1_visible_container = st.container()
                tab1_invisible_container = st.container(height=0, border=False)

            self.sidebar_tabs = {"chats": tab1_visible_container, "settings": tab2}
            with tab1_visible_container:
                left, center, right = st.columns(3)
                with left:
                    # Add button to show the costs table
                    st.toggle(
                        key="toggle_show_costs",
                        label=":moneybag:",
                        help="Show estimated token usage and associated costs",
                    )
                with center:
                    # Add button to toggle voice output
                    speaking_head_in_silhouette = "\U0001F5E3"
                    st.toggle(
                        key="toggle_voice_output",
                        label=speaking_head_in_silhouette,
                        help="Allow the assistant to speak",
                        value=True,
                    )
                with right:
                    # Add button to toggle continuous voice input
                    _infinity_emoji = "\U0000221E"
                    st.toggle(
                        key="toggle_continuous_voice_input",
                        label=":microphone:",
                        help="Speak to the assistant in a continuous manner, without "
                        "clicking the microphone button to start/stop recording",
                        value=False,
                    )

                # Add button to create a new chat
                new_chat_button = st.button(label=":heavy_plus_sign:  New Chat")

                # Reopen chats from cache (if any)
                if not self.state.get("saved_chats_reloaded", False):
                    self.state["saved_chats_reloaded"] = True
                    for cache_dir_path in self.openai_client.saved_chat_cache_paths:
                        try:
                            chat = WebAppChat.from_cache(
                                cache_dir=cache_dir_path, openai_client=self.openai_client
                            )
                        except ValidationError:
                            st.warning(
                                f"Failed to load cached chat {cache_dir_path}: "
                                + "Non-supported configs.",
                                icon="⚠️",
                            )
                            continue

                        logger.debug("Init chat from cache: {}", chat.id)
                        new_page = ChatBotPage(
                            parent=self,
                            chat_obj=chat,
                            page_title=chat.metadata.get("page_title", _RecoveredChat),
                            sidebar_title=chat.metadata.get("sidebar_title"),
                        )
                        new_page.state["messages"] = chat.load_history()
                        self.add_page(page=new_page)
                    self.selected_page = next(iter(self.pages.values()), None)

                # Create a new chat upon request or if there is none yet
                if new_chat_button or not self.pages:
                    self.add_page()

            # We'l hide the webrtc input buttom because I don't know how to customise it.
            # I'll use the component "toggle_continuous_voice_input" to toggle it
            if st.session_state["toggle_continuous_voice_input"]:
                with tab1_invisible_container:
                    self.render_continuous_audio_input_widget()

        return super().render(**kwargs)

    def _build_sidebar_tabs(self):
        def toggle_change_chat_title(page):
            page.state["edit_chat_text"] = not page.state.get("edit_chat_text", False)

        def set_page_title(page):
            page.state["edit_chat_text"] = False
            title = st.session_state.get(f"edit_{page.page_id}_text_input", "").strip()
            if not title:
                return
            page.title = title
            page.sidebar_title = title
            page.chat_obj.metadata["page_title"] = title
            page.chat_obj.metadata["sidebar_title"] = title

        with self.sidebar_tabs["chats"]:
            for page in self.pages.values():
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                with col1:
                    st.button(
                        ":wastebasket:",
                        key=f"delete_{page.page_id}",
                        type="primary",
                        use_container_width=True,
                        on_click=self.remove_page,
                        kwargs={"page": page},
                        help="Delete this chat",
                    )
                with col2:
                    if page.state.get("edit_chat_text"):
                        st.text_input(
                            "Edit Chat Title",
                            value=page.sidebar_title,
                            key=f"edit_{page.page_id}_text_input",
                            on_change=set_page_title,
                            args=[page],
                        )
                    else:
                        mtime = None
                        with contextlib.suppress(FileNotFoundError):
                            mtime = page.chat_obj.context_file_path.stat().st_mtime
                            mtime = datetime.datetime.fromtimestamp(mtime)
                            mtime = mtime.replace(microsecond=0)

                        def _set_page(page):
                            """Help setting the selected page."""
                            self.selected_page = page

                        st.button(
                            label=page.sidebar_title,
                            key=f"select_{page.page_id}",
                            help=f"Latest backup: {mtime}" if mtime else None,
                            on_click=_set_page,
                            kwargs={"page": page},
                            use_container_width=True,
                            disabled=page.page_id == self.selected_page.page_id,
                        )
                with col3:
                    st.button(
                        ":pencil:",
                        key=f"edit_{page.page_id}_button",
                        use_container_width=True,
                        on_click=toggle_change_chat_title,
                        args=[page],
                        help="Edit chat title",
                    )

    def _handle_chat_configs_value_selection(self, current_chat_configs, model_fields):
        updates_to_chat_configs = {}
        for field_name, field in model_fields.items():
            extra_info = field.json_schema_extra or {}

            # Skip fields that are not allowed to be changed
            if not extra_info.get("changeable", True):
                continue

            title = field_name.replace("_", " ").title()
            choices = VoiceChatConfigs.get_allowed_values(field=field_name)
            description = VoiceChatConfigs.get_description(field=field_name)
            field_type = VoiceChatConfigs.get_type(field=field_name)

            # Check if the field is frozen and disable corresponding UI element if so
            chat_started = self.selected_page.state.get("chat_started", False)
            disable_ui_element = extra_info.get("frozen", False) and (
                chat_started
                or any(msg["role"] == "user" for msg in self.selected_page.chat_history)
            )

            # Keep track of selected values so that selectbox doesn't reset
            current_config_value = getattr(current_chat_configs, field_name)
            element_key = f"{field_name}-pg-{self.selected_page.page_id}-ui-element"
            widget_previous_value = self.get_widget_previous_value(
                element_key, default=current_config_value
            )
            if choices:
                index = None
                try:
                    index = choices.index(widget_previous_value)
                except ValueError:
                    logger.warning(
                        "Index not found for value '{}'. The present values are {}",
                        widget_previous_value,
                        choices,
                    )

                new_field_value = st.selectbox(
                    title,
                    key=element_key,
                    options=choices,
                    index=index,
                    help=description,
                    disabled=disable_ui_element,
                    on_change=self.save_widget_previous_values,
                    args=[element_key],
                )
            elif field_type == str:
                new_field_value = st.text_input(
                    title,
                    key=element_key,
                    value=widget_previous_value,
                    help=description,
                    disabled=disable_ui_element,
                    on_change=self.save_widget_previous_values,
                    args=[element_key],
                )
            elif field_type in [int, float]:
                step = 1 if field_type == int else 0.01
                bounds = [None, None]
                for item in field.metadata:
                    with contextlib.suppress(AttributeError):
                        bounds[0] = item.gt + step
                    with contextlib.suppress(AttributeError):
                        bounds[0] = item.ge
                    with contextlib.suppress(AttributeError):
                        bounds[1] = item.lt - step
                    with contextlib.suppress(AttributeError):
                        bounds[1] = item.le

                new_field_value = st.number_input(
                    title,
                    key=element_key,
                    value=widget_previous_value,
                    placeholder="OpenAI Default",
                    min_value=bounds[0],
                    max_value=bounds[1],
                    step=step,
                    help=description,
                    disabled=disable_ui_element,
                    on_change=self.save_widget_previous_values,
                    args=[element_key],
                )
            elif field_type in (list, tuple):
                prev_value = (
                    widget_previous_value
                    if isinstance(widget_previous_value, str)
                    else "\n".join(widget_previous_value)
                )
                new_field_value = st.text_area(
                    title,
                    value=prev_value.strip(),
                    key=element_key,
                    help=description,
                    disabled=disable_ui_element,
                    on_change=self.save_widget_previous_values,
                    args=[element_key],
                )
                new_field_value = tuple(new_field_value.strip().split("\n"))
            else:
                continue

            if new_field_value != current_config_value:
                updates_to_chat_configs[field_name] = new_field_value

        return updates_to_chat_configs


def _set_button_style():
    """CSS styling for the buttons in the app."""
    st.markdown(
        """
        <style>
        .stButton button[kind="primary"] {
            background-color: white;
            opacity: 0.5;
            transition: opacity 0.3s;
        }
        .stButton button[kind="primary"]:hover {
            opacity: 1;
            border-color: #f63366;
            border-width: 2px;
        }
        .stButton button[kind="secondary"]:disabled {
            border-color: #2BB5E8;
            border-width: 2px;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
