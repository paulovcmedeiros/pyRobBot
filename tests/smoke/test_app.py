from gpt_buddy_bot.app import app
from gpt_buddy_bot.chat_configs import ChatOptions


def test_app(mocker):
    mocker.patch("streamlit.session_state", {})
    mocker.patch("pickle.load", return_value=ChatOptions())
    app.run_app()
