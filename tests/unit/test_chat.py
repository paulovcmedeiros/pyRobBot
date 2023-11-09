import openai
import pytest

from gpt_buddy_bot import GeneralConstants
from gpt_buddy_bot.general_utils import CannotConnectToApiError


@pytest.mark.order(1)
@pytest.mark.no_chat_completion_create_mocking()
@pytest.mark.parametrize("user_input", ("regular-input",))
def test_testbed_doesnt_actually_connect_to_openai(default_chat, input_builtin_mocker):
    with pytest.raises(
        CannotConnectToApiError, match=default_chat._api_connection_error_msg
    ):
        try:
            default_chat.start()
        except CannotConnectToApiError:
            raise
        else:
            pytest.exit("Refuse to continue: Testbed is trying to connect to OpenAI API!")


@pytest.mark.order(2)
def test_we_are_using_tmp_cachedir():
    try:
        assert (
            GeneralConstants.PACKAGE_CACHE_DIRECTORY
            != pytest.ORIGINAL_PACKAGE_CACHE_DIRECTORY
        )

    except AssertionError:
        pytest.exit(
            "Refuse to continue: Tests attempted to use the package's real cache dir "
            + f"({GeneralConstants.PACKAGE_CACHE_DIRECTORY})!"
        )


@pytest.mark.parametrize("user_input", ("Hi!", ""), ids=("regular-input", "empty-input"))
def test_terminal_chat(default_chat, input_builtin_mocker):
    default_chat.start()
    default_chat.__del__()  # Just to trigger testing the custom del method


def test_chat_configs(default_chat, default_chat_configs):
    assert default_chat._passed_configs == default_chat_configs


@pytest.mark.no_chat_completion_create_mocking()
@pytest.mark.parametrize("user_input", ("regular-input",))
def test_request_timeout_retry(mocker, default_chat, input_builtin_mocker):
    def _mock_openai_ChatCompletion_create(*args, **kwargs):
        raise openai.error.Timeout("Mocked timeout error was not caught!")

    mocker.patch("openai.ChatCompletion.create", new=_mock_openai_ChatCompletion_create)
    mocker.patch("time.sleep")  # Don't waste time sleeping in tests
    with pytest.raises(
        CannotConnectToApiError, match=default_chat._api_connection_error_msg
    ):
        default_chat.start()
