#!/usr/bin/env python3
import contextlib
import pickle
from subprocess import run

from . import GeneralConstants
from .chat import Chat


def run_on_terminal(args):
    """Program's main routine."""
    chat = Chat(
        model=args.model,
        base_instructions=args.initial_ai_instructions,
        send_full_history=args.send_full_history,
    )
    chat.start()


def run_on_browser(args):
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
