#!/usr/bin/env python3
"""Program's entry point."""
from .argparse_wrapper import get_parsed_args


def main(argv=None):
    """Program's main routine."""
    args = get_parsed_args(argv=argv)
    args.run_command(args=args)
