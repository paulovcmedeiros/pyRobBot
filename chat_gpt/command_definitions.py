#!/usr/bin/env python3
import contextlib
import pickle
from subprocess import run

from . import GeneralConstants
from .chat import Chat


def show_accumulated_costs(args):
    """Show the accumulated costs of the chat and exit."""
    Chat.from_cli_args(cli_args=args)


def run_on_terminal(args):
    """Run the chat on the terminal."""
    Chat.from_cli_args(cli_args=args).start()


def run_on_browser(args):
    """Run the chat on the browser."""
    with open(GeneralConstants.PARSED_ARGS_FILE, "wb") as parsed_args_file:
        pickle.dump(args, parsed_args_file)
    app_path = GeneralConstants.PACKAGE_DIRECTORY / "app" / "app.py"
    with contextlib.suppress(KeyboardInterrupt):
        run(
            [
                "streamlit",
                "run",
                app_path.as_posix(),
                "--theme.base=dark",
                "--",
                GeneralConstants.PARSED_ARGS_FILE.as_posix(),
            ]
        )
