"""Utilities for creating pages in a streamlit app."""
import contextlib
import datetime
import sys
import uuid
from abc import ABC, abstractmethod
from json.decoder import JSONDecodeError
from typing import TYPE_CHECKING

import streamlit as st
from loguru import logger
from PIL import Image

from pyrobbot import GeneralConstants
from pyrobbot.chat import Chat
from pyrobbot.chat_configs import ChatOptions

if TYPE_CHECKING:
    from pyrobbot.app.multipage import MultipageChatbotApp

_AVATAR_FILES_DIR = GeneralConstants.APP_DIR / "data"
_ASSISTANT_AVATAR_FILE_PATH = _AVATAR_FILES_DIR / "assistant_avatar.png"
_USER_AVATAR_FILE_PATH = _AVATAR_FILES_DIR / "user_avatar.png"
_ASSISTANT_AVATAR_IMAGE = Image.open(_ASSISTANT_AVATAR_FILE_PATH)
_USER_AVATAR_IMAGE = Image.open(_USER_AVATAR_FILE_PATH)


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


class ChatBotPage(AppPage):
    """Implement a chatbot page in a streamlit application, inheriting from AppPage."""

    def __init__(
        self,
        parent: "MultipageChatbotApp",
        chat_obj: Chat = None,
        sidebar_title: str = "",
        page_title: str = "",
    ):
        """Initialize new instance of the ChatBotPage class with an optional Chat object.

        Args:
            parent (MultipageChatbotApp): The parent app of the page.
            chat_obj (Chat): The chat object. Defaults to None.
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
    def chat_configs(self) -> ChatOptions:
        """Return the configs used for the page's chat object."""
        if "chat_configs" not in self.state:
            try:
                chat_options_file_path = sys.argv[-1]
                self.state["chat_configs"] = ChatOptions.from_file(chat_options_file_path)
            except (FileNotFoundError, JSONDecodeError):
                logger.warning("Could not retrieve cli args. Using default chat options.")
                self.state["chat_configs"] = ChatOptions()
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
    def chat_obj(self, new_chat_obj: Chat):
        current_chat = self.state.get("chat_obj")
        if current_chat:
            current_chat.save_cache()
            new_chat_obj.id = current_chat.id
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
        for message in self.chat_history:
            role = message["role"]
            if role == "system":
                continue
            with st.chat_message(role, avatar=self.avatars.get(role)):
                with contextlib.suppress(KeyError):
                    st.caption(message["timestamp"])
                st.markdown(message["content"])

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

    def _render_chatbot_page(self):
        """Render a chatbot page.

        Adapted from:
        <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>

        """
        st.header(self.title, divider="rainbow")

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
        if prompt := st.chat_input(
            placeholder=placeholder,
            on_submit=lambda: self.state.update({"chat_started": True}),
        ):
            time_now = datetime.datetime.now().replace(microsecond=0)
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
            with st.chat_message(
                "assistant", avatar=self.avatars["assistant"]
            ), st.empty():
                st.markdown("▌")
                full_response = ""
                for chunk in self.chat_obj.respond_user_prompt(prompt):
                    full_response += chunk
                    st.markdown(full_response + "▌")
                st.caption(datetime.datetime.now().replace(microsecond=0))
                st.markdown(full_response)

            self.chat_history.append(
                {
                    "role": "assistant",
                    "name": self.chat_obj.assistant_name,
                    "content": full_response,
                }
            )

            # Reset title according to conversation initial contents
            min_history_len_for_summary = 3
            if (
                "page_title" not in self.state
                and len(self.chat_history) > min_history_len_for_summary
            ):
                with st.spinner("Working out conversation topic..."):
                    prompt = "Summarize the messages in max 4 words.\n"
                    title = "".join(
                        self.chat_obj.respond_system_prompt(prompt, add_to_history=False)
                    )
                    self.chat_obj.metadata["page_title"] = title
                    self.chat_obj.metadata["sidebar_title"] = title
                    self.chat_obj.save_cache()

                    self.title = title
                    self.sidebar_title = title
                    st.header(title, divider="rainbow")

    def render(self):
        """Render the app's chatbot or costs page, depending on user choice."""
        if st.session_state.get("toggle_show_costs"):
            self.render_cost_estimate_page()
        else:
            self._render_chatbot_page()
