#!/usr/bin/env python3
"""Program's entry point."""
from .argparse_wrapper import get_parsed_args
from .chat_gpt import chat_with_context, simple_chat


def main(argv=None):
    """Program's main routine."""
    args = get_parsed_args(argv=argv)
    # args.intial_ai_instructions += " In your answer, include the total number of tokens used in the question/answer pair."
    chat_with_context(args)
    # simple_chat(args)


if __name__ == "__main__":
    main()
