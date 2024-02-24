from pyrobbot.app import app


def test_app(mocker, default_voice_chat_configs):
    mocker.patch("streamlit.session_state", {})
    mocker.patch("streamlit.chat_input", return_value="foobar")
    mocker.patch(
        "pyrobbot.chat_configs.VoiceChatConfigs.from_file",
        return_value=default_voice_chat_configs,
    )
    app.run_app()
