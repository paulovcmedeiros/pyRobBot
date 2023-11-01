#!/usr/bin/env python3
"""Wrappers for argparse functionality."""
import argparse
import sys

from .command_definitions import accounting, run_on_browser, run_on_terminal


def get_parsed_args(argv=None, default_command="browser"):
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

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "initial_ai_instructions",
        type=str,
        default="You answer using the minimum possible number of tokens.",
        help="Initial instructions for the AI",
        nargs="?",
    )
    common_parser.add_argument(
        "--model",
        type=lambda x: str(x).lower(),
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4"],
        help="OpenAI API engine to use for completion",
    )
    common_parser.add_argument(
        "--context-model",
        type=lambda x: None if str(x).lower() == "none" else str(x).lower(),
        default="text-embedding-ada-002",
        choices=["text-embedding-ada-002", None],
        help="OpenAI API engine to use for embedding",
    )
    common_parser.add_argument("--skip-reporting-costs", action="store_true")

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

    parser_browser = subparsers.add_parser(
        "browser", parents=[common_parser], help="Run the chat on the browser."
    )
    parser_browser.set_defaults(run_command=run_on_browser)

    parser_terminal = subparsers.add_parser(
        "terminal", parents=[common_parser], help="Run the chat on the terminal."
    )
    parser_terminal.set_defaults(run_command=run_on_terminal)

    parser_accounting = subparsers.add_parser(
        "accounting",
        aliases=["acc"],
        parents=[common_parser],
        help="Show the number of tokens used for each message.",
    )
    parser_accounting.set_defaults(run_command=accounting)

    return main_parser.parse_args(argv)
