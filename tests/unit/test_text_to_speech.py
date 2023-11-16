import contextlib

import pytest
from pydantic import ValidationError
from sounddevice import PortAudioError

from pyrobbot.chat_configs import VoiceChatConfigs
from pyrobbot.text_to_speech import VoiceChat


def test_cannot_instanciate_assistant_is_soundcard_not_imported(mocker):
    """Test that the voice chat cannot be instantiated if soundcard is not imported."""
    mocker.patch("pyrobbot.text_to_speech._sounddevice_imported", False)
    with pytest.raises(ImportError, match="Module `sounddevice`"):
        VoiceChat()


@pytest.mark.parametrize("param_name", ["sample_rate", "frame_duration"])
def test_cannot_instanciate_assistant_with_invalid_webrtcvad_params(param_name):
    """Test that the voice chat cannot be instantiated with invalid webrtcvad params."""
    with pytest.raises(ValidationError, match="Input should be"):
        VoiceChat(configs=VoiceChatConfigs(**{param_name: 1}))


def test_speak(default_voice_chat):
    """Test the speak method."""
    default_voice_chat.speak("Hello world!")


def test_listen(default_voice_chat):
    """Test the listen method."""
    with contextlib.suppress(PortAudioError):
        default_voice_chat.listen()
