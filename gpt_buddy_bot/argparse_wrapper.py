#!/usr/bin/env python3
"""Wrappers for argparse functionality."""
import argparse
import sys
from collections.abc import Sequence

from . import GeneralConstants
from .chat_configs import ChatOptions
from .command_definitions import accounting, run_on_terminal, run_on_ui


def get_parsed_args(argv=None, default_command="ui"):
    """Get parsed command line arguments.

    Args:
        argv (list): A list of passed command line args.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = [default_command]

    chat_options_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )
    argarse2pydantic = {
        "type": ChatOptions.get_type,
        "default": ChatOptions.get_default,
        "choices": ChatOptions.get_allowed_values,
        "help": ChatOptions.get_description,
    }
    for field_name in ChatOptions.model_fields:
        args_opts = {
            key: argarse2pydantic[key](field_name)
            for key in argarse2pydantic
            if argarse2pydantic[key](field_name) is not None
        }
        if "help" in args_opts:
            args_opts["help"] = f"{args_opts['help']} (default: %(default)s)"
        if "default" in args_opts and isinstance(args_opts["default"], (list, tuple)):
            args_opts.pop("type", None)
            args_opts["nargs"] = "*"

        chat_options_parser.add_argument(f"--{field_name.replace('_', '-')}", **args_opts)

    chat_options_parser.add_argument("--skip-reporting-costs", action="store_true")

    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    main_parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"{GeneralConstants.PACKAGE_NAME} v" + GeneralConstants.VERSION,
    )

    # Configure the main parser to handle the commands
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

    parser_ui = subparsers.add_parser(
        "ui",
        aliases=["app"],
        parents=[chat_options_parser],
        help="Run the chat UI on the browser.",
    )
    parser_ui.set_defaults(run_command=run_on_ui)

    parser_terminal = subparsers.add_parser(
        "terminal",
        aliases=["."],
        parents=[chat_options_parser],
        help="Run the chat on the terminal.",
    )
    parser_terminal.set_defaults(run_command=run_on_terminal)

    parser_accounting = subparsers.add_parser(
        "accounting",
        aliases=["acc"],
        help="Show the estimated number of used tokens and associated costs, and exit.",
    )
    parser_accounting.set_defaults(run_command=accounting)

    return main_parser.parse_args(argv)
