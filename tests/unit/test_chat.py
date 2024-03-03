import openai
import pytest

from pyrobbot import GeneralDefinitions
from pyrobbot.chat import Chat
from pyrobbot.chat_configs import ChatOptions


@pytest.mark.order(1)
@pytest.mark.usefixtures("_input_builtin_mocker")
@pytest.mark.no_chat_completion_create_mocking()
@pytest.mark.parametrize("user_input", ["regular-input"])
def testbed_doesnt_actually_connect_to_openai(caplog):
    llm = ChatOptions.get_allowed_values("model")[0]
    context_model = ChatOptions.get_allowed_values("context_model")[0]
    chat_configs = ChatOptions(model=llm, context_model=context_model)
    chat = Chat(configs=chat_configs)

    chat.start()
    success = chat.response_failure_message().content in caplog.text

    err_msg = "Refuse to continue: Testbed is trying to connect to OpenAI API!"
    err_msg += f"\nThis is what the logger says:\n{caplog.text}"
    if not success:
        pytest.exit(err_msg)


@pytest.mark.order(2)
def test_we_are_using_tmp_cachedir():
    try:
        assert (
            pytest.original_package_cache_directory
            != GeneralDefinitions.PACKAGE_CACHE_DIRECTORY
        )

    except AssertionError:
        pytest.exit(
            "Refuse to continue: Tests attempted to use the package's real cache dir "
            + f"({GeneralDefinitions.PACKAGE_CACHE_DIRECTORY})!"
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


@pytest.mark.usefixtures("_input_builtin_mocker")
@pytest.mark.parametrize("user_input", ["regular-input"])
def test_internet_search_can_be_triggered(default_chat, mocker):
    mocker.patch(
        "pyrobbot.openai_utils.make_api_chat_completion_call", return_value=iter(["yes"])
    )
    mocker.patch("pyrobbot.chat.Chat.respond_system_prompt", return_value=iter(["yes"]))
    mocker.patch(
        "pyrobbot.internet_utils.raw_websearch",
        return_value=iter(
            [
                {
                    "href": "foo/bar",
                    "summary": 50 * "foo ",
                    "detailed": 50 * "foo ",
                    "relevance": 1.0,
                }
            ]
        ),
    )
    default_chat.start()
