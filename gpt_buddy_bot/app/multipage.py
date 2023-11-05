"Code for the creation streamlit apps with dynamically created pages."
import contextlib

import streamlit as st
from app_page_templates import AppPage, ChatBotPage

from gpt_buddy_bot.chat import Chat
from gpt_buddy_bot.chat_configs import ChatOptions


class MultiPageApp:
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
    def pages(self) -> AppPage:
        """Return the pages of the app."""
        if "available_pages" not in st.session_state:
            st.session_state["available_pages"] = {}
        return st.session_state["available_pages"]

    def add_page(self, page: AppPage, selected: bool = False) -> None:
        """Add a page to the app."""
        self.pages[page.page_id] = page
        self.n_created_pages += 1
        if selected:
            self.register_selected_page(page)

    def remove_page(self, page: AppPage):
        """Remove a page from the app."""
        del self.pages[page.page_id]
        self.register_selected_page(next(iter(self.pages.values())))

    def register_selected_page(self, page: AppPage):
        """Register a page as selected."""
        st.session_state["selected_page"] = page

    @property
    def selected_page(self) -> ChatBotPage:
        """Return the selected page."""
        if "selected_page" not in st.session_state:
            return next(iter(self.pages.values()))
        return st.session_state["selected_page"]

    def handle_ui_page_selection(self, sidebar_tabs: dict):
        """Control page selection in the UI sidebar."""
        with sidebar_tabs["chats"]:
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
        with sidebar_tabs["settings"]:
            current_chat_configs = self.selected_page.chat_obj.configs
            updates_to_chat_configs = {}
            for field_name, field in ChatOptions.model_fields.items():
                title = field_name.replace("_", " ").title()
                choices = ChatOptions.get_allowed_values(field=field_name)
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
                    st.selectbox(title, choices, key=element_key, index=index)
                elif field_type == str:
                    st.text_input(title, value=last_field_value, key=element_key)
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

                    st.number_input(
                        title,
                        value=last_field_value,
                        placeholder="OpenAI Default",
                        min_value=bounds[0],
                        max_value=bounds[1],
                        step=step,
                        key=element_key,
                    )
                else:
                    continue

                new_field_value = st.session_state.get(element_key)
                if new_field_value != last_field_value:
                    updates_to_chat_configs[field_name] = new_field_value

        if updates_to_chat_configs:
            new_chat_configs = current_chat_configs.model_dump()
            new_chat_configs.update(updates_to_chat_configs)
            self.selected_page.chat_obj = Chat.from_dict(new_chat_configs)

    def render(self, sidebar_tabs: dict):
        """Render the multipage app with focus on the selected page."""
        self.handle_ui_page_selection(sidebar_tabs=sidebar_tabs)
        self.selected_page.render()
        st.session_state["last_rendered_page"] = self.selected_page.page_id
