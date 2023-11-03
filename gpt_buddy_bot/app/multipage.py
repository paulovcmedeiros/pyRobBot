"Code for the creation streamlit apps with dynamically created pages."
from functools import partial

import streamlit as st
from app_page_templates import AppPage


class MultiPageApp:
    """Framework for creating streamlite multipage apps.

    Adapted from:
    <https://towardsdatascience.com/
     creating-multipage-applications-using-streamlit-efficiently-b58a58134030>.

    """

    def __init__(self, sidebar_mode="buttons", **kwargs) -> None:
        """Initialise streamlit page configs."""
        st.set_page_config(**kwargs)
        self.sidebar_mode = sidebar_mode.lower()

    @property
    def pages(self) -> AppPage:
        """Return the pages of the app."""
        if "available_pages" not in st.session_state:
            st.session_state["available_pages"] = {}
        return st.session_state["available_pages"]

    def add_page(self, page: AppPage, selected: bool = False) -> None:
        """Add a page to the app."""
        self.pages[page.page_id] = page
        if selected:
            self.register_selected_page(page)

    def register_selected_page(self, page: AppPage):
        """Register a page as selected."""
        st.session_state["selected_page"] = page

    @property
    def selected_page(self) -> AppPage:
        """Return the selected page."""
        if "selected_page" not in st.session_state:
            return next(iter(self.pages.values()))
        return st.session_state["selected_page"]

    def handle_ui_page_selection(self):
        """Control page selection in the UI sidebar."""
        if self.sidebar_mode == "buttons":
            with st.sidebar:
                for page in self.pages.values():
                    st.button(
                        label=page.sidebar_title,
                        on_click=partial(self.register_selected_page, page),
                    )
        elif self.sidebar_mode == "dropdown":
            if page := st.sidebar.selectbox(
                label="Select Chat",
                options=self.pages.values(),
                format_func=lambda page: page.sidebar_title,
                index=len(self.pages) - 1,
            ):
                self.register_selected_page(page)
        else:
            raise NotImplementedError(
                f"Sidebar mode '{self.sidebar_mode}' is not implemented."
            )

    def render(self):
        """Render the multipage app with focus on the selected page."""
        self.handle_ui_page_selection()
        self.selected_page.render()
