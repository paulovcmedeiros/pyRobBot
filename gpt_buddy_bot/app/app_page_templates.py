"""Utilities for creating pages in a streamlit app."""
import pickle
import sys
import uuid
from abc import ABC, abstractmethod

import streamlit as st
from PIL import Image

from gpt_buddy_bot import GeneralConstants
from gpt_buddy_bot.chat import CannotConnectToApiError, Chat
from gpt_buddy_bot.chat_configs import ChatOptions

_AVATAR_FILES_DIR = GeneralConstants.APP_DIR / "data"
_ASSISTANT_AVATAR_FILE_PATH = _AVATAR_FILES_DIR / "assistant_avatar.png"
_USER_AVATAR_FILE_PATH = _AVATAR_FILES_DIR / "user_avatar.png"
_ASSISTANT_AVATAR_IMAGE = Image.open(_ASSISTANT_AVATAR_FILE_PATH)
_USER_AVATAR_IMAGE = Image.open(_USER_AVATAR_FILE_PATH)


# Sentinel object for when a chat is recovered from cache
_RecoveredChat = object()


class AppPage(ABC):
    """Abstract base class for pages in a streamlit app."""

    def __init__(self, sidebar_title: str = "", page_title: str = ""):
        self.page_id = str(uuid.uuid4())
        self.page_number = st.session_state.get("n_created_pages", 0) + 1

        chat_number_for_title = f"### Chat #{self.page_number}"
        if page_title is _RecoveredChat:
            self._fallback_page_title = f"{chat_number_for_title.strip('#')} (Recovered)"
            page_title = None
        else:
            self._fallback_page_title = (
                f"{GeneralConstants.APP_NAME} :speech_balloon:\n{chat_number_for_title}"
            )
            if page_title:
                self.title = page_title

        self._fallback_sidebar_title = (
            page_title if page_title else self._fallback_page_title
        )
        if sidebar_title:
            self.sidebar_title = sidebar_title

    @property
    def state(self):
        """Return the state of the page, for persistence of data."""
        if self.page_id not in st.session_state:
            st.session_state[self.page_id] = {}
        return st.session_state[self.page_id]

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
        return self.state.get("page_title", self._fallback_page_title)

    @title.setter
    def title(self, value: str):
        """Set the title of the page."""
        self.state["page_title"] = value

    @abstractmethod
    def render(self):
        """Create the page."""


class ChatBotPage(AppPage):
    def __init__(
        self, chat_obj: Chat = None, sidebar_title: str = "", page_title: str = ""
    ):
        super().__init__(sidebar_title=sidebar_title, page_title=page_title)

        if chat_obj:
            self.chat_obj = chat_obj

        self.avatars = {"assistant": _ASSISTANT_AVATAR_IMAGE, "user": _USER_AVATAR_IMAGE}

    @property
    def chat_configs(self) -> ChatOptions:
        """Return the configs used for the page's chat object."""
        if "chat_configs" not in self.state:
            chat_options_file_path = sys.argv[-1]
            with open(chat_options_file_path, "rb") as chat_configs_file:
                self.state["chat_configs"] = pickle.load(chat_configs_file)
        return self.state["chat_configs"]

    @chat_configs.setter
    def chat_configs(self, value: ChatOptions):
        self.state["chat_configs"] = ChatOptions.model_validate(value)
        if "chat_obj" in self.state:
            del self.state["chat_obj"]

    @property
    def chat_obj(self) -> Chat:
        """Return the chat object responsible for the queries in this page."""
        if "chat_obj" not in self.state:
            self.chat_obj = Chat(self.chat_configs)
        return self.state["chat_obj"]

    @chat_obj.setter
    def chat_obj(self, value: Chat):
        self.state["chat_obj"] = value
        self.state["chat_configs"] = value.configs

    @property
    def chat_history(self) -> list[dict[str, str]]:
        """Return the chat history of the page."""
        if "messages" not in self.state:
            self.state["messages"] = []
        return self.state["messages"]

    def render_chat_history(self):
        """Render the chat history of the page. Do not include system messages."""
        for message in self.chat_history:
            role = message["role"]
            if role == "system":
                continue
            with st.chat_message(role, avatar=self.avatars.get(role)):
                st.markdown(message["content"])

    def render(self):
        """Render a chatbot page.

        Adapted from:
        <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>

        """
        st.title(self.title)
        st.divider()

        if self.chat_history:
            self.render_chat_history()
        else:
            with st.chat_message("assistant", avatar=self.avatars["assistant"]):
                st.markdown(self.chat_obj.initial_greeting)
                self.chat_history.append(
                    {
                        "role": "assistant",
                        "name": self.chat_obj.assistant_name,
                        "content": self.chat_obj.initial_greeting,
                    }
                )

        # Accept user input
        placeholder = (
            f"Send a message to {self.chat_obj.assistant_name} ({self.chat_obj.model})"
        )
        if prompt := st.chat_input(placeholder=placeholder):
            # Display user message in chat message container
            with st.chat_message("user", avatar=self.avatars["user"]):
                st.markdown(prompt)
            self.chat_history.append(
                {"role": "user", "name": self.chat_obj.username, "content": prompt}
            )

            # Display (stream) assistant response in chat message container
            with st.chat_message("assistant", avatar=self.avatars["assistant"]):
                with st.empty():
                    st.markdown("▌")
                    full_response = ""
                    try:
                        for chunk in self.chat_obj.respond_user_prompt(prompt):
                            full_response += chunk
                            st.markdown(full_response + "▌")
                    except CannotConnectToApiError:
                        full_response = self.chat_obj._auth_error_msg
                    finally:
                        st.markdown(full_response)

            self.chat_history.append(
                {
                    "role": "assistant",
                    "name": self.chat_obj.assistant_name,
                    "content": full_response,
                }
            )

            # Reset title according to conversation initial contents
            if "page_title" not in self.state and len(self.chat_history) > 3:
                with st.spinner("Working out conversation topic..."):
                    prompt = "Summarize the messages in max 4 words.\n"
                    self.title = "".join(
                        self.chat_obj.respond_system_prompt(prompt, add_to_history=False)
                    )
                    self.sidebar_title = self.title
                    st.title(self.title)

                    self.chat_obj.metadata["page_title"] = self.title
                    self.chat_obj.metadata["sidebar_title"] = self.sidebar_title
