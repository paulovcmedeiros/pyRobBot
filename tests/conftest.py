from unittest.mock import MagicMock

import lorem
import numpy as np
import pygame
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

import pyrobbot
from pyrobbot.chat import Chat
from pyrobbot.chat_configs import ChatOptions
from pyrobbot.text_to_speech import LiveAssistant


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
        pyrobbot.GeneralConstants.package_cache_directory
    )


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    # Make sure we don't consume our tokens in tests
    monkeypatch.setenv("OPENAI_API_KEY", "INVALID_API_KEY")


@pytest.fixture(autouse=True)
def _mocked_general_constants(tmp_path, mocker):
    mocker.patch(
        "pyrobbot.GeneralDefinitions.package_cache_directory", tmp_path / "cache"
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

    def _mock_openai_embedding_create(*args, **kwargs):  # noqa: ARG001
        """Mock `openai.Embedding.create`. Yield from lorem ipsum instead."""
        EmbeddingRequest = type("EmbeddingRequest", (), {})
        Embedding = type("Embedding", (), {})
        Usage = type("Usage", (), {})

        embedding = Embedding()
        embedding.embedding = np.random.rand(512).tolist()
        embedding_request = EmbeddingRequest()
        embedding_request.data = [embedding]

        usage = Usage()
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


@pytest.fixture(params=ChatOptions.get_allowed_values("model"))
def llm_model(request):
    return request.param


@pytest.fixture(params=ChatOptions.get_allowed_values("context_model"))
def context_model(request):
    return request.param


@pytest.fixture()
def default_chat_configs(llm_model, context_model):
    return ChatOptions(model=llm_model, context_model=context_model)


@pytest.fixture()
def cli_args_overrides(default_chat_configs):
    args = []
    for field, value in default_chat_configs.model_dump().items():
        if value is not None:
            args = [*args, *[f"--{field.replace('_', '-')}", str(value)]]
    return args


@pytest.fixture()
def default_chat(default_chat_configs):
    return Chat(configs=default_chat_configs)


@pytest.fixture(autouse=True)
def _text_to_speech_mockers(mocker):
    """Mockers for the text-to-speech module."""
    mocker.patch(
        "pyrobbot.text_to_speech.LiveAssistant.still_talking", return_value=False
    )
    mocker.patch("gtts.gTTS.write_to_fp")

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

    mocker.patch("webrtcvad.Vad.is_speech", return_value=False)
    mocker.patch("pygame.mixer.init")
