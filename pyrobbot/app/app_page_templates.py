"""Utilities for creating pages in a streamlit app."""

import base64
import contextlib
import datetime
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import streamlit as st
from audiorecorder import audiorecorder
from PIL import Image
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

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
            self.chat_obj = chat_obj

        self.avatars = {"assistant": _ASSISTANT_AVATAR_IMAGE, "user": _USER_AVATAR_IMAGE}

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
            with right:
                audio = audiorecorder(
                    start_prompt="Rec",
                    stop_prompt="Stop",
                    pause_prompt="",
                    key=f"audiorecorder_widget_{self.page_id}",
                )
                recorded_prompt = None
                if audio.duration_seconds > min_audio_duration_seconds:
                    recorded_prompt = self.chat_obj.stt(audio).text

        return text_prompt or recorded_prompt

    def _render_chatbot_page(self):
        """Render a chatbot page.

        Adapted from:
        <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>

        """
        self.chat_obj.reply_only_as_text = not self.voice_output

        title_container = st.empty()
        title_container.header(self.title, divider="rainbow")
        chat_msgs_container = st.container(height=600, border=False)
        prompt = self.get_chat_input()

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
                    with st.empty():
                        st.markdown("▌")
                        full_response = ""
                        # When the chat object answers the user's question, it will
                        # put the response in the tts queue, then in the play speech
                        # queue, assynchronously
                        for chunk in self.chat_obj.answer_question(prompt):
                            full_response += chunk.content
                            st.markdown(full_response + "▌")
                        st.caption(datetime.datetime.now().replace(microsecond=0))
                        st.markdown(full_response)

                    full_audio = AudioSegment.silent(duration=0)
                    if self.voice_output:
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
                        self.render_custom_audio_player(
                            full_audio,
                            parent_element=audio_placeholder_container,
                            autoplay=False,
                        )

                prompt = None

                self.chat_history.append(
                    {
                        "role": "assistant",
                        "name": self.chat_obj.assistant_name,
                        "content": full_response,
                        "assistant_reply_audio_file": full_audio,
                    }
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

        # Need to delete the audiorecorder widget from the session state to prevent the
        # previous audio from being used as input again
        del st.session_state[f"audiorecorder_widget_{self.page_id}"]

    def render(self):
        """Render the app's chatbot or costs page, depending on user choice."""
        if st.session_state.get("toggle_show_costs"):
            self.render_cost_estimate_page()
        else:
            self._render_chatbot_page()
