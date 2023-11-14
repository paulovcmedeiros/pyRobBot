import contextlib

from sounddevice import PortAudioError

from pyrobbot.text_to_speech import LiveAssistant


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
