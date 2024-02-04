import contextlib

import pytest
from pydantic import ValidationError
from pydub import AudioSegment
from sounddevice import PortAudioError

from pyrobbot.chat_configs import VoiceChatConfigs
from pyrobbot.sst_and_tts import SpeechToText
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


@pytest.mark.parametrize("stt_engine", ["google", "openai"])
def test_stt(default_voice_chat, stt_engine):
    """Test the speech-to-text method."""
    default_voice_chat.stt_engine = stt_engine
    stt = SpeechToText(
        speech=AudioSegment.silent(duration=100),
        engine=stt_engine,
        general_token_usage_db=default_voice_chat.general_token_usage_db,
        token_usage_db=default_voice_chat.token_usage_db,
    )
    assert stt.text == "patched"
