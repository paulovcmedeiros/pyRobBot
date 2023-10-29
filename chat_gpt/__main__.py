#!/usr/bin/env python3
"""Program's entry point."""
from .argparse_wrapper import get_parsed_args
from .chat_gpt import simple_chat


def main(argv=None):
    """Program's main routine."""
    args = get_parsed_args(argv=argv)
    simple_chat(args)


if __name__ == "__main__":
    main()
