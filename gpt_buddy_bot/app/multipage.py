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
    def selected_page(self) -> AppPage:
        """Return the selected page."""
        if "selected_page" not in st.session_state:
            return next(iter(self.pages.values()))
        return st.session_state["selected_page"]

    def handle_ui_page_selection(self):
        """Control page selection in the UI sidebar."""
        with st.sidebar:
            col1, col2 = st.columns([0.75, 0.25])
            for page in self.pages.values():
                with col1:
                    st.button(
                        label=page.sidebar_title,
                        key=f"select_{page.page_id}",
                        on_click=partial(self.register_selected_page, page),
                        use_container_width=True,
                    )
                with col2:
                    st.button(
                        ":wastebasket:",
                        key=f"delete_{page.page_id}",
                        type="primary",
                        use_container_width=True,
                        on_click=partial(self.remove_page, page),
                        help="Delete this chat.",
                    )

    def render(self):
        """Render the multipage app with focus on the selected page."""
        self.handle_ui_page_selection()
        self.selected_page.render()
