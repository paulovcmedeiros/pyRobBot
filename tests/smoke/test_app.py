from pyrobbot.app import app


def test_app(mocker, default_voice_chat_configs):
    class MockAttrDict(dict):
        def __getattr__(self, attr):
            return self.get(attr, mocker.MagicMock())

        def __setattr__(self, attr, value):
            self[attr] = value

    mocker.patch("streamlit.session_state", MockAttrDict())
    mocker.patch("streamlit.chat_input", return_value="foobar")
    mocker.patch(
        "pyrobbot.chat_configs.VoiceChatConfigs.from_file",
        return_value=default_voice_chat_configs,
    )
    app.run_app()
