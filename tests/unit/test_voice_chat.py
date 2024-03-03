import contextlib

import pytest
from pydantic import ValidationError
from pydub import AudioSegment
from sounddevice import PortAudioError

from pyrobbot.chat_configs import VoiceChatConfigs
from pyrobbot.sst_and_tts import TextToSpeech
from pyrobbot.voice_chat import VoiceChat


def test_soundcard_import_check(mocker, caplog):
    """Test that the voice chat cannot be instantiated if soundcard is not imported."""
    mocker.patch("pyrobbot.voice_chat._sounddevice_imported", False)
    _ = VoiceChat(configs=VoiceChatConfigs())
    msg = "Module `sounddevice`, needed for local audio recording, is not available."
    assert msg in caplog.text


@pytest.mark.parametrize("param_name", ["sample_rate", "frame_duration"])
def test_cannot_instanciate_assistant_with_invalid_webrtcvad_params(param_name):
    """Test that the voice chat cannot be instantiated with invalid webrtcvad params."""
    with pytest.raises(ValidationError, match="Input should be"):
        VoiceChat(configs=VoiceChatConfigs(**{param_name: 1}))


def test_listen(default_voice_chat):
    """Test the listen method."""
    with contextlib.suppress(PortAudioError, pytest.PytestUnraisableExceptionWarning):
        default_voice_chat.listen()


def test_speak(default_voice_chat, mocker):
    tts = TextToSpeech(
        openai_client=default_voice_chat.openai_client,
        text="foo",
        general_token_usage_db=default_voice_chat.general_token_usage_db,
        token_usage_db=default_voice_chat.token_usage_db,
    )
    mocker.patch("pygame.mixer.Sound")
    mocker.patch("pyrobbot.voice_chat._get_lower_alphanumeric", return_value="ok cancel")
    mocker.patch(
        "pyrobbot.voice_chat.VoiceChat.listen",
        return_value=AudioSegment.silent(duration=150),
    )
    default_voice_chat.speak(tts)


def test_answer_question(default_voice_chat):
    default_voice_chat.answer_question("foo")


def test_interrupt_reply(default_voice_chat):
    default_voice_chat.interrupt_reply.set()
    default_voice_chat.questions_queue.get = lambda: None
    default_voice_chat.questions_queue.task_done = lambda: None
    default_voice_chat.start()


def test_handle_interrupt_expressions(default_voice_chat, mocker):
    mocker.patch("pyrobbot.general_utils.str2_minus_str1", return_value="cancel")
    default_voice_chat.questions_queue.get = lambda: None
    default_voice_chat.questions_queue.task_done = lambda: None
    default_voice_chat.questions_queue.answer_question = lambda _question: None
    msgs_to_compare = {
        "assistant_txt": "foo",
        "user_audio": AudioSegment.silent(duration=150),
    }
    default_voice_chat.check_for_interrupt_expressions_queue.put(msgs_to_compare)
    default_voice_chat.start()
