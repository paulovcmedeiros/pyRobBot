import contextlib

import streamlit
import streamlit_webrtc.component

from pyrobbot.app import app


def test_app(mocker, default_voice_chat_configs):
    class MockAttrDict(streamlit.runtime.state.session_state_proxy.SessionStateProxy):
        def __getattr__(self, attr):
            return self.get(attr, mocker.MagicMock())

        def __getitem__(self, key):
            with contextlib.suppress(KeyError):
                return super().__getitem__(key)
            return mocker.MagicMock()

    mocker.patch.object(streamlit, "session_state", new=MockAttrDict())
    mocker.patch.object(
        streamlit.runtime.state.session_state_proxy,
        "SessionStateProxy",
        new=MockAttrDict,
    )
    mocker.patch("streamlit.chat_input", return_value="foobar")
    mocker.patch(
        "pyrobbot.chat_configs.VoiceChatConfigs.from_file",
        return_value=default_voice_chat_configs,
    )
    mocker.patch.object(
        streamlit_webrtc.component,
        "webrtc_streamer",
        mocker.MagicMock(return_value=mocker.MagicMock()),
    )

    mocker.patch("streamlit.number_input", return_value=0)

    mocker.patch(
        "pyrobbot.chat_configs.VoiceChatConfigs.model_validate",
        return_value=default_voice_chat_configs,
    )

    app.run_app()
