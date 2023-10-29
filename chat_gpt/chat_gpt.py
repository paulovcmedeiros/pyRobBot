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


class BaseChatContext:
    def add_user_input(self, conversation, user_input):
        user_input_msg_obj = {"role": "user", "content": user_input}
        conversation.append(user_input_msg_obj)
        return conversation

    def add_chat_reply(self, conversation, chat_reply):
        reply_msg_obj = {"role": "assistant", "content": chat_reply}
        conversation.append(reply_msg_obj)
        return conversation


class EmbeddingBasedChatContext(BaseChatContext):
    """Chat context."""

    def __init__(self):
        self.context_file_path = GeneralConstants.EMBEDDINGS_FILE

    def add_user_input(self, conversation, user_input):
        user_input_msg_obj = {"role": "user", "content": user_input}
        store_message_to_file(
            msg_obj=user_input_msg_obj, file_path=self.context_file_path
        )
        intial_ai_instruct_msg = conversation[0]
        last_msg_exchange = conversation[-2:] if len(conversation) > 2 else []
        current_context = find_context(file_path=self.context_file_path, option="both")
        conversation = [
            intial_ai_instruct_msg,
            *current_context,
            *last_msg_exchange,
            user_input_msg_obj,
        ]
        return conversation

    def add_chat_reply(self, conversation, chat_reply):
        reply_msg_obj = {"role": "assistant", "content": chat_reply}
        conversation.append(reply_msg_obj)
        store_message_to_file(file_path=self.context_file_path, msg_obj=reply_msg_obj)
        return conversation


def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


def make_query(conversation: list, model: str):
    success = False
    print("=========================================")
    for line in conversation:
        print(line)
    print("=========================================")
    while not success:
        try:
            for line in openai.ChatCompletion.create(
                model=model,
                messages=conversation,
                request_timeout=30,
                stream=True,
            ):
                reply_content_token = getattr(line.choices[0].delta, "content", "")
                yield reply_content_token
                success = True
        except (
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
        ) as error:
            print(f"    > {error}. Retrying...")


def _base_chat(args, context):
    TOTAL_N_TOKENS = 0
    conversation = [{"role": "system", "content": args.intial_ai_instructions}]
    try:
        # while True:
        #    messages.append({"role": "user", "content": input("You: ")})
        for question in Path("questions.txt").read_text().split("\n"):
            question = question.strip()
            if not question:
                continue
            print(question)

            # Add context to the conversation
            conversation = context.add_user_input(
                conversation=conversation, user_input=question
            )

            print("AI: ", end="")
            full_reply_content = ""
            for token in make_query(conversation=conversation, model=args.model):
                print(token, end="")
                full_reply_content += token
            print("\n")

            # Update context with the reply
            conversation = context.add_chat_reply(
                conversation=conversation, chat_reply=full_reply_content
            )

            TOTAL_N_TOKENS += num_tokens_from_string(
                string="".join(msg["content"] for msg in conversation), model=args.model
            )
    except KeyboardInterrupt:
        print("Exiting.")
    print("TOTAL N TOKENS: ", TOTAL_N_TOKENS)


def simple_chat(args):
    return _base_chat(args, context=BaseChatContext())


def chat_with_context(args):
    return _base_chat(args, context=EmbeddingBasedChatContext())


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
            + "+ Previous messages.\n"
            + "Only answer next message.",
        }
    ]
