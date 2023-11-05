#!/usr/bin/env python3
import pickle
from subprocess import run

from . import GeneralConstants
from .chat import Chat
from .chat_configs import ChatOptions


def accounting(args):
    """Show the accumulated costs of the chat and exit."""
    Chat.from_cli_args(cli_args=args).report_token_usage(current_chat=False)


def run_on_terminal(args):
    """Run the chat on the terminal."""
    Chat.from_cli_args(cli_args=args).start()


def run_on_ui(args):
    """Run the chat on the browser."""
    with open(GeneralConstants.PARSED_ARGS_FILE, "wb") as chat_options_file:
        pickle.dump(ChatOptions.from_cli_args(args), chat_options_file)

    app_path = GeneralConstants.PACKAGE_DIRECTORY / "app" / "app.py"
    try:
        run(
            [
                "streamlit",
                "run",
                app_path.as_posix(),
                "--",
                GeneralConstants.PARSED_ARGS_FILE.as_posix(),
            ]
        )
    except (KeyboardInterrupt, EOFError):
        print("Exiting.")
