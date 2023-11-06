"""Entrypoint for the package's UI."""
from gpt_buddy_bot import GeneralConstants
from gpt_buddy_bot.app.multipage import MultipageChatbotApp


def run_app():
    """Create and run an instance of the pacage's app."""
    MultipageChatbotApp(
        page_title=GeneralConstants.APP_NAME, page_icon=":speech_balloon:"
    ).render()


if __name__ == "__main__":
    run_app()
