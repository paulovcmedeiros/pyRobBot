"""Utilities for creating pages in a streamlit app."""
import contextlib
import pickle
import sys
import uuid
from abc import ABC, abstractmethod

import streamlit as st

from gpt_buddy_bot import GeneralConstants
from gpt_buddy_bot.chat import Chat
from gpt_buddy_bot.chat_configs import ChatOptions


class AppPage(ABC):
    """Abstract base class for pages in a streamlit app."""

    def __init__(self, sidebar_title: str = "", page_title: str = ""):
        self.page_id = str(uuid.uuid4())
        self.page_number = st.session_state.get("n_created_pages", 0) + 1
        self._initial_sidebar_title = (
            sidebar_title if sidebar_title else f"Page {self.page_number}"
        )
        self._init_page_title = page_title if page_title else self.default_page_title

    @property
    def default_page_title(self):
        """Return the default page title."""
        return self.sidebar_title

    @property
    def state(self):
        """Return the state of the page, for persistence of data."""
        if self.page_id not in st.session_state:
            st.session_state[self.page_id] = {}
        return st.session_state[self.page_id]

    @property
    def title(self):
        """Get the title of the page."""
        return self.state.get("page_title", self._init_page_title)

    @title.setter
    def title(self, value):
        """Set the title of the page."""
        st.title(value)
        self.state["page_title"] = value
        self.state["sidebar_title"] = value

    @property
    def sidebar_title(self):
        """Get the title of the page in the sidebar."""
        return self.state.get("sidebar_title", self._initial_sidebar_title)

    @abstractmethod
    def render(self):
        """Create the page."""


class ChatBotPage(AppPage):
    @property
    def default_page_title(self):
        """Return the default page title."""
        page_title = f":speech_balloon:  {GeneralConstants.APP_NAME}\n"
        page_title += f"## {self.chat_obj.model}\n### {self.sidebar_title}"
        return page_title

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
        with contextlib.suppress(KeyError):
            del self.state["chat_obj"]

    @property
    def chat_obj(self) -> Chat:
        """Return the chat object responsible for the queries in this page."""
        if "chat_obj" not in self.state:
            self.state["chat_obj"] = Chat(self.chat_configs)
        return self.state["chat_obj"]

    @property
    def chat_history(self) -> list[dict[str, str]]:
        """Return the chat history of the page."""
        if "messages" not in self.state:
            self.state["messages"] = []
        return self.state["messages"]

    def render_chat_history(self):
        """Render the chat history of the page."""
        for message in self.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def render(self):
        """Render a chatbot page.

        Adapted from:
        <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>

        """
        st.title(self.title)

        self.render_chat_history()

        # Accept user input
        if prompt := st.chat_input("Send a message"):
            self.chat_history.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                # Use blinking cursor to indicate activity
                message_placeholder = st.empty()
                message_placeholder.markdown("▌")
                full_response = ""
                # Stream assistant response
                for chunk in self.chat_obj.respond_user_prompt(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            self.chat_history.append({"role": "assistant", "content": full_response})

            # Reset title according to conversation initial contents
            if "page_title" not in self.state and len(self.chat_history) > 3:
                with st.spinner("Working out conversation topic..."):
                    prompt = "Summarize the following msg exchange in max 4 words:\n"
                    prompt += "\n\x1f".join(
                        message["content"] for message in self.chat_history
                    )
                    self.title = "".join(self.chat_obj.respond_system_prompt(prompt))
