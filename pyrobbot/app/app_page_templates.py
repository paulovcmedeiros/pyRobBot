"""Utilities for creating pages in a streamlit app."""

import base64
import contextlib
import datetime
import queue
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Union

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from loguru import logger
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from streamlit_mic_recorder import mic_recorder

from pyrobbot.chat_configs import VoiceChatConfigs

from .app_utils import (
    AsyncReplier,
    WebAppChat,
    filter_page_info_from_queue,
    get_avatar_images,
    load_chime,
)

if TYPE_CHECKING:
    from .multipage import MultipageChatbotApp

# Sentinel object for when a chat is recovered from cache
_RecoveredChat = object()


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
        red_square = "\U0001F7E5"
        microphone = "\U0001F3A4"
        play_button = "\U000025B6"

        recording = mic_recorder(
            key=f"audiorecorder_widget_{self.page_id}",
            start_prompt=play_button + microphone,
            stop_prompt=red_square,
            just_once=True,
            use_container_width=True,
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
                <source src="data:audio/mpeg;base64,{b64}#" type="audio/mpeg">
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

        self.avatars = get_avatar_images()

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
                    if role == "assistant":
                        st.caption(message["chat_model"])
                    else:
                        st.caption(message["timestamp"])
                st.markdown(message["content"])
                with contextlib.suppress(KeyError):
                    if audio := message.get("reply_audio_file_path"):
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

    def play_chime(self, chime_type: str = "success", parent_element=None):
        """Sound a chime to send notificatons to the user."""
        chime = load_chime(chime_type)
        self.render_custom_audio_player(
            chime, hidden=True, autoplay=True, parent_element=parent_element
        )

    def render_title(self):
        """Render the title of the chatbot page."""
        with st.container(height=145, border=False):
            self.title_container = st.empty()
            self.title_container.subheader(self.title, divider="rainbow")
            left, _ = st.columns([0.7, 0.3])
            with left:
                self.status_msg_container = st.empty()

    @property
    def direct_text_prompt(self):
        """Render chat inut widgets and return the user's input."""
        placeholder = (
            f"Send a message to {self.chat_obj.assistant_name} ({self.chat_obj.model})"
        )
        text_from_manual_audio_recorder = ""
        with st.container():
            left, right = st.columns([0.9, 0.1])
            with left:
                text_from_chat_input_widget = st.chat_input(placeholder=placeholder)
            with right:
                if not st.session_state.get("toggle_continuous_voice_input"):
                    audio = self.manual_switch_mic_recorder()
                    text_from_manual_audio_recorder = self.chat_obj.stt(audio).text

        return text_from_chat_input_widget or text_from_manual_audio_recorder

    @property
    def continuous_text_prompt(self):
        """Wait until a promp from the continuous stream is ready and return it."""
        if not st.session_state.get("toggle_continuous_voice_input"):
            return None

        if not self.parent.continuous_audio_input_engine_is_running:
            logger.warning("Continuous audio input engine is not running!!!")
            self.status_msg_container.error(
                "The continuous audio input engine is not running!!!"
            )
            return None

        logger.debug("Running on continuous audio prompt. Waiting user input...")
        with self.status_msg_container:
            self.play_chime(chime_type="warning")
            with st.spinner(f"{self.chat_obj.assistant_name} is listening..."):
                while True:
                    with self.parent.text_prompt_queue.mutex:
                        this_page_prompt_queue = filter_page_info_from_queue(
                            app_page=self, the_queue=self.parent.text_prompt_queue
                        )
                    with contextlib.suppress(queue.Empty):
                        if prompt := this_page_prompt_queue.get_nowait()["text"]:
                            this_page_prompt_queue.task_done()
                            break
                    logger.trace("Still waiting for user text prompt...")
                    time.sleep(0.1)

        logger.debug("Done getting user input: {}", prompt)
        return prompt

    def _render_chatbot_page(self):  # noqa: PLR0915
        """Render a chatbot page.

        Adapted from:
        <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>

        """
        self.chat_obj.reply_only_as_text = not self.voice_output

        self.render_title()
        chat_msgs_container = st.container(height=550, border=False)
        with chat_msgs_container:
            self.render_chat_history()

        # The inputs should be rendered after the chat history. There is a performance
        # penalty  otherwise, as rendering the history causes streamlit to rerun the
        # entire page
        direct_text_prompt = self.direct_text_prompt
        continuous_stt_prompt = "" if direct_text_prompt else self.continuous_text_prompt
        prompt = direct_text_prompt or continuous_stt_prompt

        if prompt:
            logger.opt(colors=True).debug("<yellow>Recived prompt: {}</yellow>", prompt)
            self.parent.reply_ongoing.set()

            if continuous_stt_prompt:
                self.play_chime("success")
                self.status_msg_container.success("Got your message!")
                time.sleep(0.5)
        elif continuous_stt_prompt:
            self.status_msg_container.warning(
                "Could not understand your message. Please try again."
            )
            logger.opt(colors=True).debug("<yellow>Received empty prompt</yellow>")
            self.parent.reply_ongoing.clear()

        if prompt:
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
                        # Process text and audio replies asynchronously
                        replier = AsyncReplier(self, prompt)
                        reply = replier.stream_text_and_audio_reply()
                        self.chat_history.append(
                            {
                                "role": "assistant",
                                "name": self.chat_obj.assistant_name,
                                "content": reply["text"],
                                "reply_audio_file_path": reply["audio"],
                                "chat_model": self.chat_obj.model,
                            }
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
                        self.title_container.header(title, divider="rainbow")

                    # Clear the prompt queue for this page, to remove old prompts
                    with self.parent.continuous_user_prompt_queue.mutex:
                        filter_page_info_from_queue(
                            app_page=self,
                            the_queue=self.parent.continuous_user_prompt_queue,
                        )
                    with self.parent.text_prompt_queue.mutex:
                        filter_page_info_from_queue(
                            app_page=self, the_queue=self.parent.text_prompt_queue
                        )

                    replier.join()
                    self.parent.reply_ongoing.clear()

        if continuous_stt_prompt and not self.parent.reply_ongoing.is_set():
            logger.opt(colors=True).debug(
                "<yellow>Rerunning the app to wait for new input...</yellow>"
            )
            st.rerun()

    def render(self):
        """Render the app's chatbot or costs page, depending on user choice."""

        def _trim_page_padding():
            md = """
                <style>
                    .block-container {
                        padding-top: 0rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
                </style>
                """
            st.markdown(md, unsafe_allow_html=True)

        _trim_page_padding()
        if st.session_state.get("toggle_show_costs"):
            self.render_cost_estimate_page()
        else:
            self._render_chatbot_page()
        logger.debug("Reached the end of the chatbot page.")
