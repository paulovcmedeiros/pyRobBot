import contextlib

import pytest
from pydantic import ValidationError
from sounddevice import PortAudioError

from pyrobbot.chat_configs import VoiceChatConfigs
from pyrobbot.voice_chat import VoiceChat


def test_cannot_instanciate_assistant_is_soundcard_not_imported(mocker):
    """Test that the voice chat cannot be instantiated if soundcard is not imported."""
    mocker.patch("pyrobbot.voice_chat._sounddevice_imported", False)
    with pytest.raises(ImportError, match="Module `sounddevice`"):
        VoiceChat()


@pytest.mark.parametrize("param_name", ["sample_rate", "frame_duration"])
def test_cannot_instanciate_assistant_with_invalid_webrtcvad_params(param_name):
    """Test that the voice chat cannot be instantiated with invalid webrtcvad params."""
    with pytest.raises(ValidationError, match="Input should be"):
        VoiceChat(configs=VoiceChatConfigs(**{param_name: 1}))


def test_listen(default_voice_chat):
    """Test the listen method."""
    with contextlib.suppress(PortAudioError, pytest.PytestUnraisableExceptionWarning):
        default_voice_chat.listen()


def test_answer_question(default_voice_chat):
    default_voice_chat.answer_question("foo")
