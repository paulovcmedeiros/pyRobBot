import pytest

from gpt_buddy_bot.__main__ import main
from gpt_buddy_bot.argparse_wrapper import get_parsed_args


@pytest.mark.parametrize("user_input", ("Hi!", ""), ids=("regular-input", "empty-input"))
def test_terminal_command(input_builtin_mocker):
    args = ["terminal", "--report-accounting-when-done"]
    main(args)


def test_accounting_command():
    args = ["accounting"]
    main(args)


def test_default_command(mocker):
    def _mock_subprocess_run(*args, **kwargs):
        raise KeyboardInterrupt("Mocked KeyboardInterrupt")

    args = get_parsed_args(argv=[])
    assert args.command == "ui"

    mocker.patch("subprocess.run", new=_mock_subprocess_run)
    main(argv=[])
