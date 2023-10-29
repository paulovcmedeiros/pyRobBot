#!/usr/bin/env python3
import openai


def simple_chat(args):
    try:
        messages = [{"role": "system", "content": args.intial_ai_instructions}]
        while True:
            messages.append({"role": "user", "content": input("You: ")})
            query_result = openai.ChatCompletion.create(
                messages=messages, model=args.model
            )
            response_msg = query_result["choices"][0]["message"]
            messages.append(response_msg)
            print(f"AI: {response_msg['content']}\n")
    except KeyboardInterrupt:
        print("Exiting.")
