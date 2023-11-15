import contextlib

import pytest
from sounddevice import PortAudioError

from pyrobbot.text_to_speech import LiveAssistant


def test_cannot_instanciate_assistant_if_soundcard_not_imported(mocker, default_chat):
    """Test that LiveAssistant cannot be instantiated if soundcard is not imported."""
    mocker.patch("pyrobbot.text_to_speech._sounddevice_imported", False)
    with pytest.raises(ImportError, match="Module `sounddevice`"):
        LiveAssistant()


@pytest.mark.parametrize("param_name", ["sample_rate", "frame_duration"])
def test_cannot_instanciate_assistant_with_invalid_webrtcvad_params(param_name):
    """Test that LiveAssistant cannot be instantiated with invalid webrtcvad params."""
    with pytest.raises(ValueError, match=f"{param_name} must be one of:"):
        LiveAssistant(**{param_name: 1})


def test_speak():
    """Test the speak method."""
    assistant = LiveAssistant()

    # Call the speak method
    assistant.speak("Hello world!")


def test_listen():
    """Test the listen method."""
    assistant = LiveAssistant(inactivity_timeout_seconds=1e-5)

    with contextlib.suppress(PortAudioError):
        assistant.listen()
