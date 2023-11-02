"Code for the creation streamlit apps with dynamically created pages."
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
    def pages(self) -> AppPage:
        """Return the pages of the app."""
        if "available_pages" not in st.session_state:
            st.session_state["available_pages"] = {}
        return st.session_state["available_pages"]

    def add_page(self, page: AppPage) -> None:
        """Add a page to the app."""
        self.pages[page.page_id] = page
        st.session_state["switch_page"] = True

    def run(self):
        """Run the app."""
        # Drodown menu to select the page to run
        if page := st.sidebar.selectbox(
            label="Select Chat",
            options=self.pages.values(),
            format_func=lambda page: page.sidebar_title,
            index=len(self.pages) - 1,
        ):
            page.create()
