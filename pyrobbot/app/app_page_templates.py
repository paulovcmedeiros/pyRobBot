"""Utilities for creating pages in a streamlit app."""

import base64
import contextlib
import datetime
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Union

import streamlit as st
import webrtcvad
from audio_recorder_streamlit import audio_recorder
from loguru import logger
from PIL import Image
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_mic_recorder import mic_recorder

from pyrobbot import GeneralDefinitions
from pyrobbot.chat_configs import VoiceChatConfigs
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
        audio: Union[AudioSegment, str, Path, None],
        parent_element=None,
        autoplay: bool = True,
        hidden=False,
    ):
        """Autoplay an audio segment in the streamlit app."""
        # Adaped from: <https://discuss.streamlit.io/t/
        #    how-to-play-an-audio-file-automatically-generated-using-text-to-speech-
        #    in-streamlit/33201/2>

        if audio is None:
            logger.debug("No audio to play. Not rendering audio player.")
            return

        if isinstance(audio, (str, Path)):
            audio = AudioSegment.from_file(audio, format="mp3")
        elif not isinstance(audio, AudioSegment):
            raise TypeError(f"Invalid type for audio: {type(audio)}")

        autoplay = "autoplay" if autoplay else ""
        hidden = "hidden" if hidden else ""

        data = audio.export(format="mp3").read()
        b64 = base64.b64encode(data).decode()
        md = f"""
                <audio controls {autoplay} {hidden} preload="metadata">
                <source src="data:audio/mp3;base64,{b64}#" type="audio/mp3">
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
        self.text_prompt_queue = queue.Queue()

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

    def play_chime(self, chime_type: str = "correct-answer-tone", parent_element=None):
        """Sound a chime to send notificatons to the user."""
        chime = _load_chime(chime_type)
        self.render_custom_audio_player(
            chime, hidden=True, autoplay=True, parent_element=parent_element
        )

    def render_chat_input_widgets(self):
        """Render chat inut widgets and return the user's input."""
        placeholder = (
            f"Send a message to {self.chat_obj.assistant_name} ({self.chat_obj.model})"
        )
        with st.container():
            left, right = st.columns([0.95, 0.05])
            with left:
                if text_prompt := st.chat_input(
                    placeholder=placeholder, key=f"text_input_widget_{self.page_id}"
                ):
                    self.text_prompt_queue.put(text_prompt)
                    return

            with right:
                continuous_audio = st.session_state.get(
                    "toggle_continuous_voice_input", False
                )
                continuous_audio = True  # TEST

                audio = AudioSegment.empty()
                if continuous_audio:
                    # We won't handle this here. It is handled in listen, ..., sst threads
                    if not self.parent.listen_thread.is_alive():
                        raise ValueError("The listen thread is not alive")
                else:
                    audio = self.manual_switch_mic_recorder()
                    if audio and (
                        audio.duration_seconds > self.chat_obj.min_speech_duration_seconds
                    ):
                        self.text_prompt_queue.put(self.chat_obj.stt(audio).text)

    def _render_chatbot_page(self):  # noqa: PLR0915
        """Render a chatbot page.

        Adapted from:
        <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>

        """
        self.chat_obj.reply_only_as_text = not self.voice_output
        question_answer_chunks_queue = queue.Queue()
        partial_audios_queue = queue.Queue()

        with st.container(height=75, border=False):
            title_container = st.empty()
        with st.container(height=55, border=False):
            left, _ = st.columns([0.7, 0.3])
            with left:
                status_msg_container = st.empty()
        title_container.header(self.title, divider="rainbow")
        chat_msgs_container = st.container(height=600, border=False)
        with chat_msgs_container:
            self.render_chat_history()
        self.render_chat_input_widgets()

        with status_msg_container:
            logger.debug("Waiting for user text prompt...")
            self.play_chime()
            with st.spinner(f"{self.chat_obj.assistant_name} is listening..."):
                while True:
                    with contextlib.suppress(queue.Empty):
                        if prompt := self.parent.text_prompt_queue.get_nowait():
                            with self.parent.text_prompt_queue.mutex:
                                self.parent.text_prompt_queue.queue.clear()
                            break
                    with contextlib.suppress(queue.Empty):
                        if prompt := self.text_prompt_queue.get_nowait():
                            break
                    logger.trace("Still waiting for user text prompt...")
                    time.sleep(0.1)

        if prompt := prompt.strip():
            self.play_chime("option-select")
            status_msg_container.success("Got your message!")
            self.parent.reply_ongoing.set()
            logger.opt(colors=True).debug("<yellow>Recived prompt: {}</yellow>", prompt)
            question_answer_chunks_queue.queue.clear()
            partial_audios_queue.queue.clear()
        else:
            status_msg_container.warning(
                "Could not understand your message. Please try again."
            )
            logger.opt(colors=True).debug("<yellow>Received empty prompt</yellow>")
            self.parent.reply_ongoing.clear()

        with chat_msgs_container:
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
                    text_reply_container = st.empty()
                    audio_reply_container = st.empty()

                    # Create threads to process text and audio replies asynchronously
                    answer_question_thread = threading.Thread(
                        target=_put_chat_reply_chunks_in_queue,
                        args=(self.chat_obj, prompt, question_answer_chunks_queue),
                    )
                    play_partial_audios_thread = threading.Thread(
                        target=_play_queued_audios,
                        args=(
                            partial_audios_queue,
                            self.render_custom_audio_player,
                            audio_reply_container,
                        ),
                        daemon=False,
                    )
                    for thread in (answer_question_thread, play_partial_audios_thread):
                        add_script_run_ctx(thread)
                        thread.start()

                    # Render the reply
                    chunk = ""
                    full_response = ""
                    current_audio = AudioSegment.empty()
                    text_reply_container.markdown("▌")
                    status_msg_container.empty()
                    while (chunk is not None) or (current_audio is not None):
                        logger.trace("Waiting for text or audio chunks...")
                        # Render text
                        with contextlib.suppress(queue.Empty):
                            chunk = question_answer_chunks_queue.get_nowait()
                            if chunk is not None:
                                full_response += chunk
                                text_reply_container.markdown(full_response + "▌")
                            question_answer_chunks_queue.task_done()

                        # Render audio (if any)
                        with contextlib.suppress(queue.Empty):
                            current_audio = self.chat_obj.play_speech_queue.get_nowait()
                            self.chat_obj.play_speech_queue.task_done()
                            if current_audio is None:
                                partial_audios_queue.put(None)
                            else:
                                partial_audios_queue.put(current_audio.speech)

                    logger.opt(colors=True).debug(
                        "<yellow>Replied to user prompt '{}': {}</yellow>",
                        prompt,
                        full_response,
                    )
                    text_reply_container.caption(
                        datetime.datetime.now().replace(microsecond=0)
                    )
                    text_reply_container.markdown(full_response)

                    while play_partial_audios_thread.is_alive():
                        logger.trace("Waiting for partial audios to finish playing...")
                        time.sleep(0.1)

                    logger.debug("Getting path to full audio file...")
                    try:
                        full_audio_fpath = self.chat_obj.last_answer_full_audio_fpath.get(
                            timeout=2
                        )
                    except queue.Empty:
                        full_audio_fpath = None
                        logger.warning("Problem getting path to full audio file")
                    else:
                        logger.debug("Got path to full audio file: {}", full_audio_fpath)
                        self.chat_obj.last_answer_full_audio_fpath.task_done()

                    self.chat_history.append(
                        {
                            "role": "assistant",
                            "name": self.chat_obj.assistant_name,
                            "content": full_response,
                            "assistant_reply_audio_file": full_audio_fpath,
                        }
                    )

                    if full_audio_fpath:
                        self.render_custom_audio_player(
                            full_audio_fpath,
                            parent_element=audio_reply_container,
                            autoplay=False,
                        )

                # Reset title according to conversation initial contents
                min_history_len_for_summary = 3
                if (
                    "page_title" not in self.state
                    and len(self.chat_history) > min_history_len_for_summary
                ):
                    logger.debug("Working out conversation topic...")
                    prompt = "Summarize the previous messages in max 4 words"
                    title = "".join(self.chat_obj.respond_system_prompt(prompt))
                    self.chat_obj.metadata["page_title"] = title
                    self.chat_obj.metadata["sidebar_title"] = title
                    self.chat_obj.save_cache()

                    self.title = title
                    self.sidebar_title = title
                    title_container.header(title, divider="rainbow")

                self.parent.reply_ongoing.clear()

        if not self.parent.reply_ongoing.is_set():
            logger.debug("Rerunning the app")
            st.rerun()

    def render(self):
        """Render the app's chatbot or costs page, depending on user choice."""

        def _trim_page_padding():
            st.markdown(
                """
                <style>
                    .block-container {
                        padding-top: 0rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

        _trim_page_padding()
        if st.session_state.get("toggle_show_costs"):
            self.render_cost_estimate_page()
        else:
            self._render_chatbot_page()


@st.cache_data
def _load_chime(chime_type: str) -> AudioSegment:
    """Load a chime sound from the data directory."""
    type2filename = {
        "correct-answer-tone": "mixkit-correct-answer-tone-2870.wav",
        "option-select": "mixkit-interface-option-select-2573.wav",
    }

    return AudioSegment.from_file(
        GeneralDefinitions.APP_DIR / "data" / type2filename[chime_type],
        format="wav",
    )


def _put_chat_reply_chunks_in_queue(chat_obj, prompt, question_answer_chunks_queue):
    """Get chunks of the reply to the prompt and put them in the queue."""
    for chunk in chat_obj.answer_question(prompt):
        question_answer_chunks_queue.put(chunk.content)
    question_answer_chunks_queue.put(None)


def _play_queued_audios(audios_queue, audio_player_rendering_function, parent_element):
    """Play queued audio segments."""
    logger.debug("Playing queued audios...")
    while True:
        try:
            audio = audios_queue.get()
            if audio is None:
                audios_queue.task_done()
                break

            logger.debug("Playing partial audio...")
            audio_player_rendering_function(
                audio, parent_element=parent_element, autoplay=True, hidden=True
            )
            parent_element.empty()
            audios_queue.task_done()
        except Exception as error:  # noqa: BLE001
            logger.opt(exception=True).debug("Error playing partial audio.")
            logger.error(error)
            break
    logger.debug("Done playing queued audios.")
