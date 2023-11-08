"Code for the creation streamlit apps with dynamically created pages."
import contextlib
from abc import ABC, abstractmethod

import openai
import streamlit as st

from gpt_buddy_bot import GeneralConstants
from gpt_buddy_bot.app.app_page_templates import AppPage, ChatBotPage, _RecoveredChat
from gpt_buddy_bot.chat import Chat
from gpt_buddy_bot.chat_configs import ChatOptions


class AbstractMultipageApp(ABC):
    """Framework for creating streamlite multipage apps.

    Adapted from:
    <https://towardsdatascience.com/
     creating-multipage-applications-using-streamlit-efficiently-b58a58134030>.

    """

    def __init__(self, **kwargs) -> None:
        """Initialise streamlit page configs."""
        st.set_page_config(**kwargs)

    @property
    def n_created_pages(self):
        """Return the number of pages created by the app, including deleted ones."""
        return st.session_state.get("n_created_pages", 0)

    @n_created_pages.setter
    def n_created_pages(self, value):
        st.session_state["n_created_pages"] = value

    @property
    def pages(self) -> dict[AppPage]:
        """Return the pages of the app."""
        if "available_pages" not in st.session_state:
            st.session_state["available_pages"] = {}
        return st.session_state["available_pages"]

    def add_page(self, page: AppPage, selected: bool = True):
        """Add a page to the app."""
        self.pages[page.page_id] = page
        self.n_created_pages += 1
        if selected:
            self.register_selected_page(page)

    def remove_page(self, page: AppPage):
        """Remove a page from the app."""
        try:
            self.register_selected_page(next(iter(self.pages.values())))
        except StopIteration:
            self.add_page()
        self.pages[page.page_id].chat_obj.private_mode = True
        self.pages[page.page_id].chat_obj.clear_cache()
        del self.pages[page.page_id]

    def register_selected_page(self, page: AppPage):
        """Register a page as selected."""
        st.session_state["selected_page"] = page

    @property
    def selected_page(self) -> ChatBotPage:
        """Return the selected page."""
        if "selected_page" not in st.session_state:
            return next(iter(self.pages.values()))
        return st.session_state["selected_page"]

    @abstractmethod
    def handle_ui_page_selection(self, **kwargs):
        """Control page selection in the UI sidebar."""

    def render(self, **kwargs):
        """Render the multipage app with focus on the selected page."""
        self.handle_ui_page_selection(**kwargs)
        self.selected_page.render()
        st.session_state["last_rendered_page"] = self.selected_page.page_id


class MultipageChatbotApp(AbstractMultipageApp):
    def init_openai_client(self):
        # Initialize the OpenAI API client
        placeholher = (
            "OPENAI_API_KEY detected"
            if GeneralConstants.OPENAI_API_KEY
            else "You need this to use the chat"
        )
        openai_api_key = st.text_input(
            label="OpenAI API Key (required)",
            placeholder=placeholher,
            key="openai_api_key",
            type="password",
            help="[OpenAI API auth key](https://platform.openai.com/account/api-keys)",
        )
        openai.api_key = (
            openai_api_key if openai_api_key else GeneralConstants.OPENAI_API_KEY
        )
        if not openai.api_key:
            st.write(":red[You need to provide a key to use the chat]")

    def add_page(self, page: ChatBotPage = None, selected: bool = True, **kwargs):
        if page is None:
            page = ChatBotPage(**kwargs)
        return super().add_page(page=page, selected=selected)

    def handle_ui_page_selection(self):
        """Control page selection in the UI sidebar."""
        with self.sidebar_tabs["chats"]:
            for page in self.pages.values():
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.button(
                        label=page.sidebar_title,
                        key=f"select_{page.page_id}",
                        on_click=self.register_selected_page,
                        kwargs={"page": page},
                        use_container_width=True,
                    )
                with col2:
                    st.button(
                        ":wastebasket:",
                        key=f"delete_{page.page_id}",
                        type="primary",
                        use_container_width=True,
                        on_click=self.remove_page,
                        kwargs={"page": page},
                        help="Delete this chat.",
                    )

        with self.sidebar_tabs["settings"]:
            caption = f"\u2699\uFE0F Settings for Chat #{self.selected_page.page_number}"
            if self.selected_page.title != self.selected_page._fallback_page_title:
                caption += f": {self.selected_page.title}"
            st.caption(caption)
            current_chat_configs = self.selected_page.chat_obj.configs
            updates_to_chat_configs = {}

            # Present the user with the model and instructions fields first
            field_names = ["model", "ai_instructions"]
            field_names += [field_name for field_name in ChatOptions.model_fields]
            field_names = list(dict.fromkeys(field_names))
            model_fiedls = {k: ChatOptions.model_fields[k] for k in field_names}

            for field_name, field in model_fiedls.items():
                title = field_name.replace("_", " ").title()
                choices = ChatOptions.get_allowed_values(field=field_name)
                description = ChatOptions.get_description(field=field_name)
                field_type = ChatOptions.get_type(field=field_name)

                element_key = f"{field_name}-pg-{self.selected_page.page_id}-ui-element"
                last_field_value = getattr(current_chat_configs, field_name)
                if choices:
                    index = (
                        0
                        if st.session_state.get("last_rendered_page")
                        == self.selected_page.page_id
                        else choices.index(last_field_value)
                    )
                    new_field_value = st.selectbox(
                        title, choices, key=element_key, index=index, help=description
                    )
                elif field_type == str:
                    new_field_value = st.text_input(
                        title, value=last_field_value, key=element_key, help=description
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
                        value=last_field_value,
                        placeholder="OpenAI Default",
                        min_value=bounds[0],
                        max_value=bounds[1],
                        step=step,
                        key=element_key,
                        help=description,
                    )
                elif field_type in (list, tuple):
                    new_field_value = st.text_area(
                        title,
                        value="\n".join(last_field_value),
                        key=element_key,
                        help=description,
                    )
                    new_field_value = tuple(new_field_value.split("\n"))
                else:
                    continue

                if new_field_value != last_field_value:
                    updates_to_chat_configs[field_name] = new_field_value

        if updates_to_chat_configs:
            new_chat_configs = current_chat_configs.model_dump()
            new_chat_configs.update(updates_to_chat_configs)
            self.selected_page.chat_obj = Chat.from_dict(new_chat_configs)

    def get_saved_chat_cache_dir_paths(self):
        """Get the filepaths of saved chat contexts, sorted by last modified."""
        return sorted(
            (
                directory
                for directory in GeneralConstants.CHAT_CACHE_DIR.glob("chat_*/")
                if next(directory.iterdir(), False)
            ),
            key=lambda fpath: fpath.stat().st_mtime,
            reverse=True,
        )

    def render(self, **kwargs):
        with st.sidebar:
            st.title(GeneralConstants.APP_NAME)
            self.init_openai_client()
            # Create a sidebar with tabs for chats and settings
            tab1, tab2 = st.tabs(["Chats", "Settings for Current Chat"])
            self.sidebar_tabs = {"chats": tab1, "settings": tab2}
            with tab1:
                # Add button to create a new chat
                new_chat_button = st.button(label=":heavy_plus_sign:  New Chat")

                # Reopen chats from cache (if any)
                if not st.session_state.get("saved_chats_reloaded", False):
                    st.session_state["saved_chats_reloaded"] = True
                    for cache_dir_path in self.get_saved_chat_cache_dir_paths():
                        chat = Chat.from_cache(cache_dir=cache_dir_path)
                        new_page = ChatBotPage(
                            chat_obj=chat,
                            page_title=chat.metadata.get("page_title", _RecoveredChat),
                            sidebar_title=chat.metadata.get("sidebar_title"),
                        )
                        new_page.state["messages"] = chat.load_history()
                        self.add_page(page=new_page)
                    self.register_selected_page(next(iter(self.pages.values()), None))

                # Create a new chat upon request or if there is none yet
                if new_chat_button or not self.pages:
                    self.add_page()

        return super().render(**kwargs)
