import contextlib
from unittest.mock import MagicMock

import pygame
from pyrobbot.text_to_speech import LiveAssistant


def test_speak(mocker):
    """Test the speak method."""
    mocker.patch(
        "pyrobbot.text_to_speech.LiveAssistant.still_talking", return_value=False
    )
    mocker.patch("gtts.gTTS.write_to_fp", return_value=b"\x00\x00\x00\x00")

    orig_func = LiveAssistant.sound_from_bytes_io

    def mock_sound_from_bytes_io(self: LiveAssistant, bytes_io):
        try:
            return orig_func(self, bytes_io)
        except pygame.error:
            return MagicMock()

    mocker.patch(
        "pyrobbot.text_to_speech.LiveAssistant.sound_from_bytes_io",
        mock_sound_from_bytes_io,
    )

    assistant = LiveAssistant()

    # Call the speak method
    assistant.speak("Hello world!")
