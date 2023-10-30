#!/usr/bin/env python3
"""Program's entry point."""
from .argparse_wrapper import get_parsed_args
from .chat_gpt import Chat


def main(argv=None):
    """Program's main routine."""
    args = get_parsed_args(argv=argv)
    chat = Chat(
        model=args.model,
        base_instructions=args.initial_ai_instructions,
        send_full_history=args.send_full_history,
    )
    chat.start()


if __name__ == "__main__":
    main()
