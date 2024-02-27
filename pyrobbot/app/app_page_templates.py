"""Utilities for creating pages in a streamlit app."""

import base64
import contextlib
import datetime
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import streamlit as st
import webrtcvad
from audio_recorder_streamlit import audio_recorder
from loguru import logger
from PIL import Image
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_mic_recorder import mic_recorder
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from pyrobbot import GeneralDefinitions
from pyrobbot.chat_configs import VoiceChatConfigs
from pyrobbot.general_utils import trim_silence
from pyrobbot.voice_chat import VoiceChat

if TYPE_CHECKING:

    from pyrobbot.app.multipage import MultipageChatbotApp

_AVATAR_FILES_DIR = GeneralDefinitions.APP_DIR / "data"
_ASSISTANT_AVATAR_FILE_PATH = _AVATAR_FILES_DIR / "assistant_avatar.png"
_USER_AVATAR_FILE_PATH = _AVATAR_FILES_DIR / "user_avatar.png"
_ASSISTANT_AVATAR_IMAGE = Image.open(_ASSISTANT_AVATAR_FILE_PATH)
_USER_AVATAR_IMAGE = Image.open(_USER_AVATAR_FILE_PATH)


# Sentinel object for when a chat is recovered from cache
_RecoveredChat = object()


class WebAppChat(VoiceChat):
    """A chat object for web apps."""

    def __init__(self, **kwargs):
        """Initialize a new instance of the WebAppChat class."""
        super().__init__(**kwargs)
        self.tts_conversion_watcher_thread.start()
        self.handle_update_audio_history_thread.start()


class AppPage(ABC):
    """Abstract base class for a page within a streamlit application."""

    def __init__(
        self, parent: "MultipageChatbotApp", sidebar_title: str = "", page_title: str = ""
    ):
        """Initializes a new instance of the AppPage class.

        Args:
            parent (MultipageChatbotApp): The parent app of the page.
            sidebar_title (str, optional): The title to be displayed in the sidebar.
                Defaults to an empty string.
            page_title (str, optional): The title to be displayed on the page.
                Defaults to an empty string.
        """
        self.page_id = str(uuid.uuid4())
        self.parent = parent
        self.page_number = self.parent.state.get("n_created_pages", 0) + 1

        chat_number_for_title = f"Chat #{self.page_number}"
        if page_title is _RecoveredChat:
            self.fallback_page_title = f"{chat_number_for_title.strip('#')} (Recovered)"
            page_title = None
        else:
            self.fallback_page_title = chat_number_for_title
            if page_title:
                self.title = page_title

        self._fallback_sidebar_title = page_title if page_title else chat_number_for_title
        if sidebar_title:
            self.sidebar_title = sidebar_title

    @property
    def state(self):
        """Return the state of the page, for persistence of data."""
        if self.page_id not in self.parent.state:
            self.parent.state[self.page_id] = {}
        return self.parent.state[self.page_id]

    @property
    def sidebar_title(self):
        """Get the title of the page in the sidebar."""
        return self.state.get("sidebar_title", self._fallback_sidebar_title)

    @sidebar_title.setter
    def sidebar_title(self, value: str):
        """Set the sidebar title for the page."""
        self.state["sidebar_title"] = value

    @property
    def title(self):
        """Get the title of the page."""
        return self.state.get("page_title", self.fallback_page_title)

    @title.setter
    def title(self, value: str):
        """Set the title of the page."""
        self.state["page_title"] = value

    @abstractmethod
    def render(self):
        """Create the page."""

    def continuous_mic_recorder(self):
        """Record audio from the microphone in a continuous loop."""
        audio_bytes = audio_recorder(
            text="", icon_size="2x", energy_threshold=-1, key=f"AR_{self.page_id}"
        )

        if audio_bytes is None:
            return AudioSegment.silent(duration=0)

        return AudioSegment(data=audio_bytes)

    def manual_switch_mic_recorder(self):
        """Record audio from the microphone."""
        studio_microphone = "\U0001F399"
        red_square = "\U0001F7E5"

        recording = mic_recorder(
            key=f"audiorecorder_widget_{self.page_id}",
            start_prompt=studio_microphone,
            stop_prompt=red_square,
            just_once=True,
            use_container_width=True,
            callback=None,
            args=(),
            kwargs={},
        )

        if recording is None:
            return AudioSegment.silent(duration=0)

        return AudioSegment(
            data=recording["bytes"],
            sample_width=recording["sample_width"],
            frame_rate=recording["sample_rate"],
            channels=1,
        )

    def render_custom_audio_player(
        self,
        audio: AudioSegment,
        parent_element=None,
        autoplay: bool = True,
        start_sec: int = 0,
    ):
        """Autoplay an audio segment in the streamlit app."""
        # Adaped from: <https://discuss.streamlit.io/t/
        #    how-to-play-an-audio-file-automatically-generated-using-text-to-speech-
        #    in-streamlit/33201/2>
        autoplay = "autoplay" if autoplay else ""
        data = audio.export(format="mp3").read()
        b64 = base64.b64encode(data).decode()
        md = f"""
                <audio controls {autoplay} preload="auto">
                <source src="data:audio/mp3;base64,{b64}#{start_sec}" type="audio/mp3">
                </audio>
                """
        parent_element = parent_element or st
        parent_element.markdown(md, unsafe_allow_html=True)
        if autoplay:
            time.sleep(audio.duration_seconds)


class ChatBotPage(AppPage):
    """Implement a chatbot page in a streamlit application, inheriting from AppPage."""

    def __init__(
        self,
        parent: "MultipageChatbotApp",
        chat_obj: WebAppChat = None,
        sidebar_title: str = "",
        page_title: str = "",
    ):
        """Initialize new instance of the ChatBotPage class with an opt WebAppChat object.

        Args:
            parent (MultipageChatbotApp): The parent app of the page.
            chat_obj (WebAppChat): The chat object. Defaults to None.
            sidebar_title (str): The sidebar title for the chatbot page.
                Defaults to an empty string.
            page_title (str): The title for the chatbot page.
                Defaults to an empty string.
        """
        super().__init__(
            parent=parent, sidebar_title=sidebar_title, page_title=page_title
        )

        if chat_obj:
            logger.debug("Setting page chat to chat with ID=<{}>", chat_obj.id)
            self.chat_obj = chat_obj
        else:
            logger.debug("ChatBotPage created wihout specific chat. Creating default.")
            _ = self.chat_obj
            logger.debug("Default chat id=<{}>", self.chat_obj.id)

        self.avatars = {"assistant": _ASSISTANT_AVATAR_IMAGE, "user": _USER_AVATAR_IMAGE}

        # Definitions related to webrtc_streamer
        self.vad = webrtcvad.Vad(2)
        self.continuous_user_prompt_queue = queue.Queue()
        self.text_prompt_queue = queue.Queue()
        self.possible_speech_chunks_queue = queue.Queue()
        self.audio_playing_chunks_queue = queue.Queue()
        self.reply_ongoing = threading.Event()
        self.incoming_frame_queue = queue.Queue()

        self.chat_for_async = self.chat_obj

        self.listen_thread = threading.Thread(
            target=self.listen,
            args=(self.chat_for_async, self.possible_speech_chunks_queue),
            daemon=True,
        )

        self.continuous_user_prompt_thread = threading.Thread(
            target=self.handle_continuous_user_prompt,
            args=(
                self.possible_speech_chunks_queue,
                self.audio_playing_chunks_queue,
                self.continuous_user_prompt_queue,
            ),
            daemon=True,
        )

        # See <https://github.com/streamlit/streamlit/issues/1326#issuecomment-1597918085>
        add_script_run_ctx(self.listen_thread)
        add_script_run_ctx(self.continuous_user_prompt_thread)

        self.listen_thread.start()
        self.continuous_user_prompt_thread.start()

    def render_continuous_audio_input_widget(self):
        """Render the continuous audio input widget."""
        # Definitions related to webrtc_streamer
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        def audio_frame_callback(frame):
            logger.trace("Received audio frame from the stream")
            self.incoming_frame_queue.put(frame)
            return frame

        self.stream_audio_context = webrtc_streamer(
            key="sendonly-audio",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"audio": True, "video": False},
            desired_playing_state=True,
            audio_frame_callback=audio_frame_callback,
        )

        return self.stream_audio_context

    def listen(self, chat_obj, possible_speech_chunks_queue):
        """Listen for speech from the browser."""
        # This deque will be employed to keep a moving window of audio chunks to monitor
        # voice activity. The length of the deque is calculated such that the concatenated
        # audio chunks will produce an audio at most inactivity_timeout_seconds long
        #
        # Mind that none of Streamlit's APIs are safe to call from any thread other than
        # the main one. See, e.g., <https://discuss.streamlit.io/t/
        # changing-session-state-not-reflecting-in-active-python-thread/37683

        logger.debug("Listening for speech from the browser...")
        moving_window_speech_likelihood = 0.0
        user_has_been_speaking = False
        audio_chunks_moving_window = deque(
            maxlen=int(
                (1000.0 * chat_obj.inactivity_timeout_seconds) / chat_obj.frame_duration
            )
        )

        while True:
            try:
                logger.trace("Waiting for audio frame from the stream...")
                received_audio_frame = self.incoming_frame_queue.get()
                logger.trace("Received audio frame from the stream")

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
                audio_chunk = audio_chunk.set_frame_rate(
                    chat_obj.sample_rate
                ).set_channels(1)

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
                moving_window_length = len(audio_chunks_moving_window)
                if moving_window_length == audio_chunks_moving_window.maxlen:
                    voice_activity = (
                        chunk["is_speech"] for chunk in audio_chunks_moving_window
                    )
                    moving_window_speech_likelihood = (
                        sum(voice_activity) / moving_window_length
                    )

                user_speaking_now = (
                    moving_window_speech_likelihood
                    >= chat_obj.speech_likelihood_threshold
                )
                if user_has_been_speaking:
                    if user_speaking_now:
                        possible_speech_chunks_queue.put(audio_chunk)
                    else:
                        logger.info("User has stopped speaking.")
                        user_has_been_speaking = False
                        possible_speech_chunks_queue.put(None)
                        continue
                elif user_speaking_now:
                    logger.info("User has started speaking.")
                    user_has_been_speaking = True
                    for past_audio_chunk in audio_chunks_moving_window:
                        possible_speech_chunks_queue.put(past_audio_chunk["audio"])

            except Exception as error:  # noqa: BLE001
                logger.error(error)
            finally:
                self.incoming_frame_queue.task_done()

    def handle_continuous_user_prompt(
        self,
        possible_speech_chunks_queue,
        audio_playing_chunks_queue,
        continuous_user_prompt_queue,
    ):
        """Play audio."""
        logger.debug("Handling continuous user prompt...")
        while True:
            try:
                logger.trace("Waiting for new speech chunk...")
                new_audio_chunk = possible_speech_chunks_queue.get()
                if self.reply_ongoing.is_set():
                    logger.debug("Reply is ongoing. Discardig audio chunk")
                    possible_speech_chunks_queue.task_done()
                    continue

                logger.trace("Processing new speech chunk")
                if new_audio_chunk is None:
                    # User has stopped speaking. Concatenate all audios from
                    # play_audio_queue and send the result to be played
                    logger.debug("Preparing audio to send as user input...")
                    concatenated_audio = AudioSegment.empty()
                    while not audio_playing_chunks_queue.empty():
                        concatenated_audio += audio_playing_chunks_queue.get()
                    audio_playing_chunks_queue.task_done()
                    continuous_user_prompt_queue.put(trim_silence(concatenated_audio))
                    logger.debug("Done preparing audio to send as user input.")
                else:
                    audio_playing_chunks_queue.put(new_audio_chunk)

                possible_speech_chunks_queue.task_done()
            except Exception as error:  # noqa: BLE001
                logger.error(error)

    @property
    def chat_configs(self) -> VoiceChatConfigs:
        """Return the configs used for the page's chat object."""
        if "chat_configs" not in self.state:
            self.state["chat_configs"] = self.parent.state["chat_configs"]
        return self.state["chat_configs"]

    @chat_configs.setter
    def chat_configs(self, value: VoiceChatConfigs):
        self.state["chat_configs"] = VoiceChatConfigs.model_validate(value)
        if "chat_obj" in self.state:
            del self.state["chat_obj"]

    @property
    def chat_obj(self) -> WebAppChat:
        """Return the chat object responsible for the queries on this page."""
        if "chat_obj" not in self.state:
            self.chat_obj = WebAppChat(
                configs=self.chat_configs, openai_client=self.parent.openai_client
            )
        return self.state["chat_obj"]

    @chat_obj.setter
    def chat_obj(self, new_chat_obj: WebAppChat):
        current_chat = self.state.get("chat_obj")
        if current_chat:
            logger.debug(
                "Copy new_chat=<{}> into current_chat=<{}>. Current chat ID kept.",
                new_chat_obj.id,
                current_chat.id,
            )
            current_chat.save_cache()
            new_chat_obj.id = current_chat.id
        new_chat_obj.openai_client = self.parent.openai_client
        self.state["chat_obj"] = new_chat_obj
        self.state["chat_configs"] = new_chat_obj.configs
        new_chat_obj.save_cache()

    @property
    def chat_history(self) -> list[dict[str, str]]:
        """Return the chat history of the page."""
        if "messages" not in self.state:
            self.state["messages"] = []
        return self.state["messages"]

    def render_chat_history(self):
        """Render the chat history of the page. Do not include system messages."""
        with st.chat_message("assistant", avatar=self.avatars["assistant"]):
            st.markdown(self.chat_obj.initial_greeting)

        for message in self.chat_history:
            role = message["role"]
            if role == "system":
                continue
            with st.chat_message(role, avatar=self.avatars.get(role)):
                with contextlib.suppress(KeyError):
                    st.caption(message["timestamp"])
                st.markdown(message["content"])
                with contextlib.suppress(KeyError):
                    if audio := message.get("assistant_reply_audio_file"):
                        with contextlib.suppress(CouldntDecodeError):
                            if not isinstance(audio, AudioSegment):
                                audio = AudioSegment.from_file(audio, format="mp3")
                            if len(audio) > 0:
                                self.render_custom_audio_player(audio, autoplay=False)

    def render_cost_estimate_page(self):
        """Render the estimated costs information in the chat."""
        general_df = self.chat_obj.general_token_usage_db.get_usage_balance_dataframe()
        chat_df = self.chat_obj.token_usage_db.get_usage_balance_dataframe()
        dfs = {"All Recorded Chats": general_df, "Current Chat": chat_df}

        st.header(dfs["Current Chat"].attrs["description"], divider="rainbow")
        with st.container():
            for category, df in dfs.items():
                st.subheader(f"**{category}**")
                st.dataframe(df)
                st.write()
            st.caption(df.attrs["disclaimer"])

    @property
    def voice_output(self) -> bool:
        """Return the state of the voice output toggle."""
        return st.session_state.get("toggle_voice_output", False)

    def get_chat_input(self):
        """Render chat inut widgets and return the user's input."""
        placeholder = (
            f"Send a message to {self.chat_obj.assistant_name} ({self.chat_obj.model})"
        )
        min_audio_duration_seconds = 0.1
        with st.container():
            left, right = st.columns([0.95, 0.05])
            with left:
                text_prompt = st.chat_input(
                    placeholder=placeholder, key=f"text_input_widget_{self.page_id}"
                )
                if text_prompt:
                    self.text_prompt_queue.put(text_prompt)
            with right:
                continuous_audio = st.session_state.get(
                    "toggle_continuous_voice_input", False
                )
                continuous_audio = True

                audio = AudioSegment.empty()
                if continuous_audio:
                    if not self.listen_thread.is_alive():
                        raise ValueError("The listen thread is not alive")

                    self.render_continuous_audio_input_widget()
                    if not self.stream_audio_context.state.playing:
                        logger.debug("Waiting for the audio stream to start...")
                        time.sleep(0.5)
                        if not text_prompt:
                            self.text_prompt_queue.put(None)
                        return

                    if self.reply_ongoing.is_set():
                        logger.debug("Waiting for the reply to finish...")
                    elif not self.continuous_user_prompt_queue.empty():
                        logger.debug("Waiting for prompt (audio) to process...")
                        audio = self.continuous_user_prompt_queue.get()
                        self.continuous_user_prompt_queue.task_done()
                        logger.debug("Got prompt (in audio) to process.")

                else:
                    audio = self.manual_switch_mic_recorder()

                recorded_prompt = ""
                if audio.duration_seconds > min_audio_duration_seconds:
                    recorded_prompt = self.chat_obj.stt(audio).text
                    self.text_prompt_queue.put(recorded_prompt)

    def _render_chatbot_page(self):  # noqa: PLR0915
        """Render a chatbot page.

        Adapted from:
        <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>

        """
        self.chat_obj.reply_only_as_text = not self.voice_output

        title_container = st.empty()
        title_container.header(self.title, divider="rainbow")
        chat_msgs_container = st.container(height=600, border=False)

        self.get_chat_input()
        logger.debug("Waiting for user text prompt...")
        try:
            prompt = self.text_prompt_queue.get(timeout=2)
            self.text_prompt_queue.task_done()
        except queue.Empty:
            prompt = None

        if prompt:
            self.reply_ongoing.set()
            logger.opt(colors=True).debug(
                "<yellow>Got prompt from user: {}</yellow>", prompt
            )

        else:
            self.reply_ongoing.clear()
            logger.opt(colors=True).debug("No prompt from user.")

        with chat_msgs_container:
            self.render_chat_history()
            # Process user input
            if prompt:
                time_now = datetime.datetime.now().replace(microsecond=0)
                self.state.update({"chat_started": True})
                # Display user message in chat message container
                with st.chat_message("user", avatar=self.avatars["user"]):
                    st.caption(time_now)
                    st.markdown(prompt)
                self.chat_history.append(
                    {
                        "role": "user",
                        "name": self.chat_obj.username,
                        "content": prompt,
                        "timestamp": time_now,
                    }
                )

                # Display (stream) assistant response in chat message container
                with st.chat_message("assistant", avatar=self.avatars["assistant"]):
                    full_response = st.write_stream(
                        item.content for item in self.chat_obj.answer_question(prompt)
                    )
                    logger.opt(colors=True).debug(
                        "<yellow>Replied to user prompt '{}': {}</yellow>",
                        prompt,
                        full_response,
                    )

                    full_audio = AudioSegment.silent(duration=0)
                    if self.voice_output:
                        # When the chat object answers the user's question, it will
                        # put the response in its tts queue and the resulting objects are
                        # then in the play speech queue (assynchronously)
                        audio_placeholder_container = st.empty()
                        while not self.chat_obj.play_speech_queue.empty():
                            current_tts = self.chat_obj.play_speech_queue.get()
                            full_audio += current_tts.speech
                            self.render_custom_audio_player(
                                current_tts.speech,
                                parent_element=audio_placeholder_container,
                            )
                            audio_placeholder_container.empty()
                            self.chat_obj.play_speech_queue.task_done()

                    self.chat_history.append(
                        {
                            "role": "assistant",
                            "name": self.chat_obj.assistant_name,
                            "content": full_response,
                            "assistant_reply_audio_file": full_audio,
                        }
                    )

                    if full_audio.duration_seconds > 0:
                        self.render_custom_audio_player(
                            full_audio,
                            parent_element=audio_placeholder_container,
                            autoplay=False,
                        )

                # Reset title according to conversation initial contents
                min_history_len_for_summary = 3
                if (
                    "page_title" not in self.state
                    and len(self.chat_history) > min_history_len_for_summary
                ):
                    with st.spinner("Working out conversation topic..."):
                        prompt = "Summarize the previous messages in max 4 words"
                        title = "".join(self.chat_obj.respond_system_prompt(prompt))
                        self.chat_obj.metadata["page_title"] = title
                        self.chat_obj.metadata["sidebar_title"] = title
                        self.chat_obj.save_cache()

                        self.title = title
                        self.sidebar_title = title
                        title_container.header(title, divider="rainbow")

        self.reply_ongoing.clear()
        st.rerun()

    def render(self):
        """Render the app's chatbot or costs page, depending on user choice."""
        if st.session_state.get("toggle_show_costs"):
            self.render_cost_estimate_page()
        else:
            self._render_chatbot_page()
