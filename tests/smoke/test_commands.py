import pytest

from gpt_buddy_bot.__main__ import main
from gpt_buddy_bot.argparse_wrapper import get_parsed_args


@pytest.mark.usefixtures("_input_builtin_mocker")
@pytest.mark.parametrize("user_input", ["Hi!", ""], ids=["regular-input", "empty-input"])
def test_terminal_command(cli_args_overrides):
    args = ["terminal", "--report-accounting-when-done", *cli_args_overrides]
    args = list(dict.fromkeys(args))
    main(args)


def test_accounting_command():
    main(["accounting"])


def test_default_command(mocker):
    def _mock_subprocess_run(*args, **kwargs):  # noqa: ARG001
        raise KeyboardInterrupt("Mocked KeyboardInterrupt")

    args = get_parsed_args(argv=[])
    assert args.command == "ui"

    mocker.patch("subprocess.run", new=_mock_subprocess_run)
    main(argv=[])
