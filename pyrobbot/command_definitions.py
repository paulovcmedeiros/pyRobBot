#!/usr/bin/env python3
"""Commands supported by the package's script."""
import subprocess

from loguru import logger

from . import GeneralConstants
from .chat import Chat
from .chat_configs import ChatOptions
from .openai_utils import CannotConnectToApiError
from .text_to_speech import VoiceChat


def voice_chat(args):
    """Run the chat on the terminal."""
    chat = VoiceChat.from_cli_args(cli_args=args)
    try:
        chat.start()
    except CannotConnectToApiError as error:
        logger.error("API connection problems: {}\nExiting.", error)
        raise SystemExit(1) from error


def browser_chat(args):
    """Run the chat on the browser."""
    ChatOptions.from_cli_args(args).export(fpath=GeneralConstants.PARSED_ARGS_FILE)
    try:
        subprocess.run(
            [  # noqa: S603, S607
                "streamlit",
                "run",
                GeneralConstants.APP_PATH.as_posix(),
                "--",
                GeneralConstants.PARSED_ARGS_FILE.as_posix(),
            ],
            cwd=GeneralConstants.APP_DIR.as_posix(),
            check=True,
        )
    except (KeyboardInterrupt, EOFError):
        logger.info("Exiting.")


def terminal_chat(args):
    """Run the chat on the terminal."""
    chat = Chat.from_cli_args(cli_args=args)
    chat.start()
    if args.report_accounting_when_done:
        chat.report_token_usage(report_general=True)


def accounting_report(args):
    """Show the accumulated costs of the chat and exit."""
    chat = Chat.from_cli_args(cli_args=args)
    # Prevent chat from creating entry in the cache directory
    chat.private_mode = True
    chat.report_token_usage(report_general=True, report_current_chat=False)
