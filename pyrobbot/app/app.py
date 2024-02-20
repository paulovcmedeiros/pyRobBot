"""Entrypoint for the package's UI."""

from pyrobbot import GeneralDefinitions
from pyrobbot.app.multipage import MultipageChatbotApp


def run_app():
    """Create and run an instance of the pacage's app."""
    MultipageChatbotApp(
        page_title=GeneralDefinitions.APP_NAME,
        page_icon=":speech_balloon:",
        layout="wide",
    ).render()


if __name__ == "__main__":
    run_app()
