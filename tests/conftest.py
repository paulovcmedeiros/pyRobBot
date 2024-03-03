import io

import lorem
import numpy as np
import pydub
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

import pyrobbot
from pyrobbot.chat import Chat
from pyrobbot.chat_configs import ChatOptions, VoiceChatConfigs
from pyrobbot.voice_chat import VoiceChat


@pytest.fixture()
def caplog(caplog: LogCaptureFixture):
    """Override the default `caplog` fixture to propagate Loguru to the caplog handler."""
    # Source: <https://loguru.readthedocs.io/en/stable/resources/migration.html
    #          #replacing-caplog-fixture-from-pytest-library>
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


# Register markers and constants
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_chat_completion_create_mocking: do not mock openai.ChatCompletion.create",
    )
    config.addinivalue_line(
        "markers",
        "no_embedding_create_mocking: mark test to not mock openai.Embedding.create",
    )

    pytest.original_package_cache_directory = (
        pyrobbot.GeneralDefinitions.PACKAGE_CACHE_DIRECTORY
    )


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    # Make sure we don't consume our tokens in tests
    monkeypatch.setenv("OPENAI_API_KEY", "INVALID_API_KEY")
    monkeypatch.setenv("STREAMLIT_SERVER_HEADLESS", "true")


@pytest.fixture(autouse=True)
def _mocked_general_constants(tmp_path, mocker):
    mocker.patch(
        "pyrobbot.GeneralDefinitions.PACKAGE_CACHE_DIRECTORY", tmp_path / "cache"
    )


@pytest.fixture()
def mock_wav_bytes_string():
    """Mock a WAV file as a bytes string."""
    return (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00"
        b"\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00"
    )


@pytest.fixture(autouse=True)
def _openai_api_request_mockers(request, mocker):
    """Mockers for OpenAI API requests. We don't want to consume our tokens in tests."""

    def _mock_openai_chat_completion_create(*args, **kwargs):  # noqa: ARG001
        """Mock `openai.ChatCompletion.create`. Yield from lorem ipsum instead."""
        completion_chunk = type("CompletionChunk", (), {})
        completion_chunk_choice = type("CompletionChunkChoice", (), {})
        completion_chunk_choice_delta = type("CompletionChunkChoiceDelta", (), {})
        for word in lorem.get_paragraph().split():
            completion_chunk_choice_delta.content = word + " "
            completion_chunk_choice.delta = completion_chunk_choice_delta
            completion_chunk.choices = [completion_chunk_choice]
            yield completion_chunk

        # Yield some code as well, to test the code filtering
        code_path = pyrobbot.GeneralDefinitions.PACKAGE_DIRECTORY / "__init__.py"
        for word in [
            "```python\n",
            *code_path.read_text().splitlines(keepends=True)[:5],
            "```\n",
        ]:
            completion_chunk_choice_delta.content = word + " "
            completion_chunk_choice.delta = completion_chunk_choice_delta
            completion_chunk.choices = [completion_chunk_choice]
            yield completion_chunk

    def _mock_openai_embedding_create(*args, **kwargs):  # noqa: ARG001
        """Mock `openai.Embedding.create`. Yield from lorem ipsum instead."""
        embedding_request_mock_type = type("EmbeddingRequest", (), {})
        embedding_mock_type = type("Embedding", (), {})
        usage_mock_type = type("Usage", (), {})

        embedding = embedding_mock_type()
        embedding.embedding = np.random.rand(512).tolist()
        embedding_request = embedding_request_mock_type()
        embedding_request.data = [embedding]

        usage = usage_mock_type()
        usage.prompt_tokens = 0
        usage.total_tokens = 0
        embedding_request.usage = usage

        return embedding_request

    if "no_chat_completion_create_mocking" not in request.keywords:
        mocker.patch(
            "openai.resources.chat.completions.Completions.create",
            new=_mock_openai_chat_completion_create,
        )
    if "no_embedding_create_mocking" not in request.keywords:
        mocker.patch(
            "openai.resources.embeddings.Embeddings.create",
            new=_mock_openai_embedding_create,
        )


@pytest.fixture(autouse=True)
def _internet_search_mockers(mocker):
    """Mockers for the internet search module."""
    mocker.patch("duckduckgo_search.DDGS.text", return_value=lorem.get_paragraph())


@pytest.fixture()
def _input_builtin_mocker(mocker, user_input):
    """Mock the `input` builtin. Raise `KeyboardInterrupt` after the second call."""

    # We allow two calls in order to allow for the chat context handler to kick in
    def _mock_input(*args, **kwargs):  # noqa: ARG001
        try:
            _mock_input.execution_counter += 1
        except AttributeError:
            _mock_input.execution_counter = 0
        if _mock_input.execution_counter > 1:
            raise KeyboardInterrupt
        return user_input

    mocker.patch(  # noqa: PT008
        "builtins.input", new=lambda _: _mock_input(user_input=user_input)
    )


@pytest.fixture(params=ChatOptions.get_allowed_values("model")[:2])
def llm_model(request):
    return request.param


context_model_values = ChatOptions.get_allowed_values("context_model")


@pytest.fixture(params=[context_model_values[0], context_model_values[2]])
def context_model(request):
    return request.param


@pytest.fixture()
def default_chat_configs(llm_model, context_model):
    return ChatOptions(model=llm_model, context_model=context_model)


@pytest.fixture()
def default_voice_chat_configs(llm_model, context_model):
    return VoiceChatConfigs(model=llm_model, context_model=context_model)


@pytest.fixture()
def cli_args_overrides(default_chat_configs):
    args = []
    for field, value in default_chat_configs.model_dump().items():
        if value not in [None, True, False]:
            args = [*args, *[f"--{field.replace('_', '-')}", str(value)]]
    return args


@pytest.fixture()
def default_chat(default_chat_configs):
    return Chat(configs=default_chat_configs)


@pytest.fixture()
def default_voice_chat(default_voice_chat_configs):
    chat = VoiceChat(configs=default_voice_chat_configs)
    chat.inactivity_timeout_seconds = 1e-5
    chat.tts_engine = "google"
    return chat


@pytest.fixture(autouse=True)
def _voice_chat_mockers(mocker, mock_wav_bytes_string):
    """Mockers for the text-to-speech module."""
    mocker.patch(
        "pyrobbot.voice_chat.VoiceChat._assistant_still_replying", return_value=False
    )

    mock_google_tts_obj = type("mock_gTTS", (), {})
    mock_openai_tts_response = type("mock_openai_tts_response", (), {})

    def _mock_iter_bytes(*args, **kwargs):  # noqa: ARG001
        return [mock_wav_bytes_string]

    mock_openai_tts_response.iter_bytes = _mock_iter_bytes

    mocker.patch(
        "pydub.AudioSegment.from_mp3",
        return_value=pydub.AudioSegment.from_wav(io.BytesIO(mock_wav_bytes_string)),
    )
    mocker.patch("gtts.gTTS", return_value=mock_google_tts_obj)
    mocker.patch(
        "openai.resources.audio.speech.Speech.create",
        return_value=mock_openai_tts_response,
    )
    mock_transcription = type("MockTranscription", (), {})
    mock_transcription.text = "patched"
    mocker.patch(
        "openai.resources.audio.transcriptions.Transcriptions.create",
        return_value=mock_transcription,
    )
    mocker.patch(
        "speech_recognition.Recognizer.recognize_google",
        return_value=mock_transcription.text,
    )

    mocker.patch("webrtcvad.Vad.is_speech", return_value=False)
    mocker.patch("pygame.mixer.init")
    mocker.patch("chime.play_wav")
    mocker.patch("chime.play_wav")
