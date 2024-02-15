import pytest
from pydub import AudioSegment

from pyrobbot.sst_and_tts import SpeechToText


@pytest.mark.parametrize("stt_engine", ["google", "openai"])
def test_stt(default_voice_chat, stt_engine):
    """Test the speech-to-text method."""
    default_voice_chat.stt_engine = stt_engine
    stt = SpeechToText(
        openai_client=default_voice_chat.openai_client,
        speech=AudioSegment.silent(duration=100),
        engine=stt_engine,
        general_token_usage_db=default_voice_chat.general_token_usage_db,
        token_usage_db=default_voice_chat.token_usage_db,
    )
    assert stt.text == "patched"
