#!/usr/bin/env python3
"""Wrappers for argparse functionality."""
import argparse
import sys


def get_parsed_args(argv=None):
    """Get parsed command line arguments.

    Args:
        argv (list): A list of passed command line args.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "intial_ai_instructions",
        type=str,
        default=(
            "You are a helpful assistant that answers correctly and concisely. "
            + "Use the fewest tokens possible unless otherwise instructed."
        ),
        help="Initial instructions for the AI",
        nargs="?",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4"],
        help="OpenAI API engine to use for completion",
    )
    return parser.parse_args(argv)
