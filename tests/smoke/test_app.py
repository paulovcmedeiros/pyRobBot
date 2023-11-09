from gpt_buddy_bot.app import app


def test_app(mocker, default_chat_configs):
    mocker.patch("streamlit.session_state", {})
    mocker.patch("pickle.load", return_value=default_chat_configs)
    app.run_app()
