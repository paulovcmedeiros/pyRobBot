import os

import lorem
import openai
import pytest

from gpt_buddy_bot.chat import Chat
from gpt_buddy_bot.chat_configs import ChatOptions


@pytest.fixture(scope="session", autouse=True)
def set_env():
    # Make sure we don't consume our tokens in tests
    os.environ["OPENAI_API_KEY"] = "INVALID_API_KEY"
    openai.api_key = os.environ["OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def openai_api_request_mockers(mocker):
    """Mockers for OpenAI API requests. We don't want to consume our tokens in tests."""

    def _mock_openai_ChatCompletion_create(*args, **kwargs):
        """Mock `openai.ChatCompletion.create`. Yield from lorem ipsum instead."""
        completion_chunk = type("CompletionChunk", (), {})
        completion_chunk_choice = type("CompletionChunkChoice", (), {})
        completion_chunk_choice_delta = type("CompletionChunkChoiceDelta", (), {})
        for word in lorem.get_paragraph().split():
            completion_chunk_choice_delta.content = word + " "
            completion_chunk_choice.delta = completion_chunk_choice_delta
            completion_chunk.choices = [completion_chunk_choice]
            yield completion_chunk

    def _mock_openai_Embedding_create(*args, **kwargs):
        """Mock `openai.Embedding.create`. Yield from lorem ipsum instead."""
        embedding_request = {
            "data": [{"embedding": [0.0] * 512}],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        return embedding_request

    mocker.patch("openai.ChatCompletion.create", new=_mock_openai_ChatCompletion_create)
    mocker.patch("openai.Embedding.create", new=_mock_openai_Embedding_create)


@pytest.fixture(autouse=True)
def input_builtin_mocker(mocker, user_input):
    """Mock the `input` builtin. Raise `KeyboardInterrupt` after the first call."""

    def _mock_input(*args, **kwargs):
        try:
            _mock_input.execution_counter += 1
        except AttributeError:
            _mock_input.execution_counter = 0
        if _mock_input.execution_counter > 0:
            raise KeyboardInterrupt
        return user_input

    mocker.patch("builtins.input", new=lambda _: _mock_input(user_input=user_input))


@pytest.fixture
def default_chat_configs():
    return ChatOptions()


@pytest.fixture()
def default_chat(default_chat_configs):
    return Chat(configs=default_chat_configs)
