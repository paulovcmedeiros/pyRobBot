import openai
import pytest

from pyrobbot import GeneralConstants
from pyrobbot.chat import Chat


@pytest.mark.order(1)
@pytest.mark.usefixtures("_input_builtin_mocker")
@pytest.mark.no_chat_completion_create_mocking()
@pytest.mark.parametrize("user_input", ["regular-input"])
def testbed_doesnt_actually_connect_to_openai(default_chat, caplog):
    default_chat.start()
    success = default_chat.response_failure_message() in caplog.text

    err_msg = "Refuse to continue: Testbed is trying to connect to OpenAI API!"
    err_msg += f"\nThis is what the logger says:\n{caplog.text}"
    if not success:
        pytest.exit(err_msg)


@pytest.mark.order(2)
def test_we_are_using_tmp_cachedir():
    try:
        assert (
            GeneralConstants.package_cache_directory
            != pytest.original_package_cache_directory
        )

    except AssertionError:
        pytest.exit(
            "Refuse to continue: Tests attempted to use the package's real cache dir "
            + f"({GeneralConstants.package_cache_directory})!"
        )


@pytest.mark.usefixtures("_input_builtin_mocker")
@pytest.mark.parametrize("user_input", ["Hi!", ""], ids=["regular-input", "empty-input"])
def test_terminal_chat(default_chat):
    default_chat.start()
    default_chat.__del__()  # Just to trigger testing the custom del method


def test_chat_configs(default_chat, default_chat_configs):
    assert default_chat._passed_configs == default_chat_configs


@pytest.mark.no_chat_completion_create_mocking()
@pytest.mark.usefixtures("_input_builtin_mocker")
@pytest.mark.parametrize("user_input", ["regular-input"])
def test_request_timeout_retry(mocker, default_chat, caplog):
    def _mock_openai_chat_completion_create(*args, **kwargs):  # noqa: ARG001
        raise openai.APITimeoutError("Mocked timeout error was not caught!")

    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        new=_mock_openai_chat_completion_create,
    )
    mocker.patch("time.sleep")  # Don't waste time sleeping in tests
    default_chat.start()
    assert "APITimeoutError" in caplog.text


def test_can_read_chat_from_cache(default_chat):
    default_chat.save_cache()
    new_chat = Chat.from_cache(default_chat.cache_dir)
    assert new_chat.configs == default_chat.configs


def test_create_from_cache_returns_default_chat_if_invalid_cachedir(default_chat, caplog):
    _ = Chat.from_cache(default_chat.cache_dir / "foobar")
    assert "Creating Chat with default configs" in caplog.text
