#!/usr/bin/env python3
"""Commands supported by the package's script."""
import subprocess

from loguru import logger

from . import GeneralDefinitions
from .chat import Chat
from .chat_configs import ChatOptions
from .voice_chat import VoiceChat


def voice_chat(args):
    """Start a voice-based chat."""
    VoiceChat.from_cli_args(cli_args=args).start()


def browser_chat(args):
    """Run the chat on the browser."""
    ChatOptions.from_cli_args(args).export(fpath=GeneralDefinitions.PARSED_ARGS_FILE)
    try:
        subprocess.run(
            [  # noqa: S603, S607
                "streamlit",
                "run",
                GeneralDefinitions.APP_PATH.as_posix(),
                "--",
                GeneralDefinitions.PARSED_ARGS_FILE.as_posix(),
            ],
            cwd=GeneralDefinitions.APP_DIR.as_posix(),
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
