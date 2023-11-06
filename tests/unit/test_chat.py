import openai
import pytest

from gpt_buddy_bot.chat import CannotConnectToApiError, Chat
from gpt_buddy_bot.chat_configs import ChatOptions


@pytest.mark.order(1)
@pytest.mark.no_chat_completion_create_mocking
@pytest.mark.parametrize("user_input", ("regular-input",))
def test_testbed_doesnt_actually_connect_to_openai(default_chat, input_builtin_mocker):
    with pytest.raises(CannotConnectToApiError, match=default_chat._auth_error_msg):
        try:
            default_chat.start()
        except CannotConnectToApiError:
            raise
        else:
            pytest.exit("Refuse to continue: Testbed is trying to connect to OpenAI API!")


@pytest.mark.parametrize("user_input", ("Hi!", ""), ids=("regular-input", "empty-input"))
def test_terminal_chat(default_chat, input_builtin_mocker):
    default_chat.start()
    default_chat.__del__()  # Just to trigger testing the custom del method


def test_chat_configs(default_chat, default_chat_configs):
    assert default_chat.configs == default_chat_configs


@pytest.mark.no_chat_completion_create_mocking
@pytest.mark.parametrize("user_input", ("regular-input",))
def test_request_timeout_retry(mocker, default_chat, input_builtin_mocker):
    def _mock_openai_ChatCompletion_create(*args, **kwargs):
        raise openai.error.Timeout("Mocked timeout error")

    mocker.patch("openai.ChatCompletion.create", new=_mock_openai_ChatCompletion_create)
    with pytest.raises(CannotConnectToApiError, match=default_chat._auth_error_msg):
        default_chat.start()


@pytest.mark.parametrize("context_model", ChatOptions.get_allowed_values("context_model"))
@pytest.mark.parametrize("user_input", ("regular-input",))
def test_chat_context_handlers(default_chat_configs, input_builtin_mocker, context_model):
    chat_configs_dict = default_chat_configs.model_dump()
    chat_configs_dict.update({"context_model": context_model})
    chat = Chat.from_dict(chat_configs_dict)
    chat.start()
