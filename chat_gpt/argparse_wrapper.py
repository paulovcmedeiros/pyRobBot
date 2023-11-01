#!/usr/bin/env python3
"""Wrappers for argparse functionality."""
import argparse
import sys

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

    chat_options_parser = argparse.ArgumentParser(add_help=False)
    chat_options_parser.add_argument(
        "initial_ai_instructions",
        type=str,
        default="You answer using the minimum possible number of tokens.",
        help="Initial instructions for the AI",
        nargs="?",
    )
    chat_options_parser.add_argument(
        "--model",
        type=lambda x: str(x).lower(),
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4"],
        help="OpenAI API engine to use for completion",
    )
    chat_options_parser.add_argument(
        "--context-model",
        type=lambda x: None if str(x).lower() == "none" else str(x).lower(),
        default="text-embedding-ada-002",
        choices=["text-embedding-ada-002", None],
        help="OpenAI API engine to use for embedding",
    )
    chat_options_parser.add_argument("--skip-reporting-costs", action="store_true")

    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
