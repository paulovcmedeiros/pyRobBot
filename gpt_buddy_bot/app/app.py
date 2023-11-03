"""Entrypoint for the package's UI."""
import streamlit as st
from app_page_templates import ChatBotPage
from multipage import MultiPageApp

from gpt_buddy_bot import GeneralConstants


def run_app():
    """Create and run an instance of the pacage's app."""
    app = MultiPageApp(page_title=GeneralConstants.APP_NAME, page_icon=":speech_balloon:")
    with st.sidebar:
        tab1, tab2 = st.tabs(["Chats", "Settings"])
        sidebar_tabs = {"chats": tab1, "settings": tab2}
        with tab1:
            # Create a new chat upon init or button press
            if st.button(label=":speech_balloon: New Chat") or not app.pages:
                app.add_page(ChatBotPage(), selected=True)
    app.render(sidebar_tabs=sidebar_tabs)


if __name__ == "__main__":
    run_app()
