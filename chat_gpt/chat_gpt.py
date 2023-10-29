#!/usr/bin/env python3
import ast
import csv
import json
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import distances_from_embeddings

from . import GeneralConstants


def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


def simple_chat(args):
    TOTAL_N_TOKENS = 0
    conversation = [{"role": "system", "content": args.intial_ai_instructions}]
    try:
        # while True:
        #    messages.append({"role": "user", "content": input("You: ")})
        for question in Path("questions.txt").read_text().split("\n"):
            question = question.strip()
            if not question:
                continue

            conversation.append({"role": "user", "content": question})
            print(question)

            success = False
            while not success:
                try:
                    query_result = openai.ChatCompletion.create(
                        messages=conversation,
                        model=args.model,
                        request_timeout=30,
                    )
                except (
                    openai.error.ServiceUnavailableError,
                    openai.error.Timeout,
                ) as error:
                    print(f"    > {error}. Retrying...")
                else:
                    success = True
            response_msg = query_result["choices"][0]["message"]
            conversation.append(response_msg)

            ai_reply = response_msg["content"]
            print(f"AI: {ai_reply}")

            text_for_token_count = "".join(msg["content"] for msg in conversation)
            n_tokens = num_tokens_from_string(
                string=text_for_token_count, model=args.model
            )
            TOTAL_N_TOKENS += n_tokens
            print("    > Total tokens used: ", n_tokens)
            print()
    except KeyboardInterrupt:
        print("Exiting.")
    print("TOTAL N TOKENS: ", TOTAL_N_TOKENS)


def store_message_to_file(
    msg_obj: dict, file_path: Path = GeneralConstants.EMBEDDINGS_FILE
):
    """Store message and embeddings to file."""
    # Adapted from <https://community.openai.com/t/
    #  use-embeddings-to-retrieve-relevant-context-for-ai-assistant/268538>
    response = openai.Embedding.create(
        model="text-embedding-ada-002", input=msg_obj["content"]
    )
    emb_mess_pair = {
        "embedding": json.dumps(response["data"][0]["embedding"]),
        "message": json.dumps(msg_obj),
    }

    init_file = not file_path.exists() or file_path.stat().st_size == 0
    write_mode = "w" if init_file else "a"

    with open(file_path, write_mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=emb_mess_pair.keys())
        if init_file:
            writer.writeheader()
        writer.writerow(emb_mess_pair)


def find_context(file_path: Path = GeneralConstants.EMBEDDINGS_FILE, option="both"):
    """Lookup context from file."""
    # Adapted from <https://community.openai.com/t/
    #  use-embeddings-to-retrieve-relevant-context-for-ai-assistant/268538>
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []

    df = pd.read_csv(file_path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    if option == "both":
        message_list_embeddings = df["embedding"].values[:-3]
    elif option == "assistant":
        message_list_embeddings = df.loc[
            df["message"].apply(lambda x: ast.literal_eval(x)["role"] == "assistant"),
            "embedding",
        ].values[-1]
    elif option == "user":
        message_list_embeddings = df.loc[
            df["message"].apply(lambda x: ast.literal_eval(x)["role"] == "user"),
            "embedding",
        ].values[:-2]
    else:
        return []  # Return an empty list if no context is found

    query_embedding = df["embedding"].values[-1]
    distances = distances_from_embeddings(
        query_embedding, message_list_embeddings, distance_metric="L1"
    )
    mask = (np.array(distances) < 21.6)[np.argsort(distances)]

    message_array = df["message"].iloc[np.argsort(distances)][mask]
    message_array = [] if message_array is None else message_array[:4]

    message_objects = [json.loads(message) for message in message_array]
    context_for_current_user_query = ""
    for msg in message_objects:
        context_for_current_user_query += f"{msg['role']}: {msg['content']}\n"

    if not context_for_current_user_query:
        return []

    return [
        {
            "role": "system",
            "content": f"Your knowledge: {context_for_current_user_query} "
            + "+ Previous messages. "
            + "Only answer next message.",
        }
    ]


def chat_with_context(
    args,
    context_file_path: Path = GeneralConstants.EMBEDDINGS_FILE,
):
    intial_ai_instruct_msg = {"role": "system", "content": args.intial_ai_instructions}
    conversation = []
    TOTAL_N_TOKENS = 0
    try:
        # while True:
        #    user_input = {"role": "user", "content": input("You: ")}
        for question in Path("questions.txt").read_text().split("\n"):
            question = question.strip()
            if not question:
                continue
            user_input = {"role": "user", "content": question}
            store_message_to_file(msg_obj=user_input, file_path=context_file_path)

            last_msg_exchange = conversation[-2:] if len(conversation) > 2 else []
            current_context = find_context(file_path=context_file_path, option="both")
            conversation = [
                intial_ai_instruct_msg,
                *last_msg_exchange,
                *current_context,
                user_input,
            ]

            print(question, end="")
            print(f" (conversation length: {len(conversation)})")

            print("AI: ", end="")
            full_reply_content = ""
            success = False
            while not success:
                try:
                    for line in openai.ChatCompletion.create(
                        model=args.model,
                        messages=conversation,
                        request_timeout=30,
                        stream=True,
                    ):
                        reply_content_token = getattr(
                            line.choices[0].delta, "content", ""
                        )
                        print(reply_content_token, end="")
                        full_reply_content += reply_content_token
                except (
                    openai.error.ServiceUnavailableError,
                    openai.error.Timeout,
                ) as error:
                    print(f"    > {error}. Retrying...")
                else:
                    success = True
            print()

            reply_msg_obj = {"role": "assistant", "content": full_reply_content}
            store_message_to_file(file_path=context_file_path, msg_obj=reply_msg_obj)
            conversation.append(reply_msg_obj)

            text_for_token_count = "".join(msg["content"] for msg in conversation)
            n_tokens = num_tokens_from_string(
                string=text_for_token_count, model=args.model
            )
            print("    > Total tokens used: ", n_tokens)
            print()
            TOTAL_N_TOKENS += n_tokens

    except KeyboardInterrupt:
        print("Exiting.")
    print("TOTAL N TOKENS: ", TOTAL_N_TOKENS)
