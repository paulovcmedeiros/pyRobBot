#!/usr/bin/env python3
"""Wrappers for argparse functionality."""
import argparse
import contextlib
import sys

from pydantic import BaseModel

from . import GeneralDefinitions
from .chat_configs import ChatOptions, VoiceChatConfigs
from .command_definitions import (
    accounting_report,
    browser_chat,
    terminal_chat,
    voice_chat,
)


def _populate_parser_from_pydantic_model(parser, model: BaseModel):
    _argarse2pydantic = {
        "type": model.get_type,
        "default": model.get_default,
        "choices": model.get_allowed_values,
        "help": model.get_description,
    }

    for field_name, field in model.model_fields.items():
        with contextlib.suppress(AttributeError):
            if not field.json_schema_extra.get("changeable", True):
                continue

        args_opts = {
            key: _argarse2pydantic[key](field_name)
            for key in _argarse2pydantic
            if _argarse2pydantic[key](field_name) is not None
        }

        if args_opts.get("type") == bool:
            if args_opts.get("default") is True:
                args_opts["action"] = "store_false"
            else:
                args_opts["action"] = "store_true"
            args_opts.pop("default", None)
            args_opts.pop("type", None)

        args_opts["required"] = field.is_required()
        if "help" in args_opts:
            args_opts["help"] = f"{args_opts['help']} (default: %(default)s)"
        if "default" in args_opts and isinstance(args_opts["default"], (list, tuple)):
            args_opts.pop("type", None)
            args_opts["nargs"] = "*"

        parser.add_argument(f"--{field_name.replace('_', '-')}", **args_opts)

    return parser


def get_parsed_args(argv=None, default_command="ui"):
    """Get parsed command line arguments.

    Args:
        argv (list): A list of passed command line args.
        default_command (str, optional): The default command to run.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    """
    if argv is None:
        argv = sys.argv[1:]
    first_argv = next(iter(argv), "'")
    info_flags = ["--version", "-v", "-h", "--help"]
    if not argv or (first_argv.startswith("-") and first_argv not in info_flags):
        argv = [default_command, *argv]

    # Main parser that will handle the script's commands
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    main_parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"{GeneralDefinitions.PACKAGE_NAME} v" + GeneralDefinitions.VERSION,
    )
    subparsers = main_parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        description=(
            "Valid commands (note that commands also accept their "
            + "own arguments, in particular [-h]):"
        ),
        help="command description",
    )

    # Common options to most commands
    chat_options_parser = _populate_parser_from_pydantic_model(
        parser=argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
        ),
        model=ChatOptions,
    )
    chat_options_parser.add_argument(
        "--report-accounting-when-done",
        action="store_true",
        help="Report estimated costs when done with the chat.",
    )

    # Web app chat
    parser_ui = subparsers.add_parser(
        "ui",
        aliases=["app", "webapp", "browser"],
        parents=[chat_options_parser],
        help="Run the chat UI on the browser.",
    )
    parser_ui.set_defaults(run_command=browser_chat)

    # Voice chat
    voice_options_parser = _populate_parser_from_pydantic_model(
        parser=argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
        ),
        model=VoiceChatConfigs,
    )
    parser_voice_chat = subparsers.add_parser(
        "voice",
        aliases=["v", "speech", "talk"],
        parents=[voice_options_parser],
        help="Run the chat over voice only.",
    )
    parser_voice_chat.set_defaults(run_command=voice_chat)

    # Terminal chat
    parser_terminal = subparsers.add_parser(
        "terminal",
        aliases=["."],
        parents=[chat_options_parser],
        help="Run the chat on the terminal.",
    )
    parser_terminal.set_defaults(run_command=terminal_chat)

    # Accounting report
    parser_accounting = subparsers.add_parser(
        "accounting",
        aliases=["acc"],
        help="Show the estimated number of used tokens and associated costs, and exit.",
    )
    parser_accounting.set_defaults(run_command=accounting_report)

    return main_parser.parse_args(argv)
