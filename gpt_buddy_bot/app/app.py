"""Entrypoint for the package's UI."""
import streamlit as st
from app_page_templates import ChatBotPage
from multipage import MultiPageApp

from gpt_buddy_bot import GeneralConstants


def run_app():
    """Create and run an instance of the pacage's app."""
    app = MultiPageApp(page_title=GeneralConstants.APP_NAME, page_icon=":speech_balloon:")
    with st.sidebar:
        # Create a new chat upon init or button press
        if st.button(label="Create New Chat") or not app.pages:
            app.add_page(ChatBotPage(sidebar_title=f"Chat {len(app.pages) + 1}"))
    app.run()


if __name__ == "__main__":
    run_app()
