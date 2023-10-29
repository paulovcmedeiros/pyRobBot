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
    def __init__(self, parent_chat):
        self.parent_chat = parent_chat

    def add_user_input(self, conversation, user_input):
        user_input_msg_obj = {
            "role": "user",
            "name": self.parent_chat.username,
            "content": user_input,
        }
        conversation.append(user_input_msg_obj)
        return conversation

    def add_chat_reply(self, conversation, chat_reply):
        reply_msg_obj = {
            "role": "assistant",
            "name": self.parent_chat.assistant_name,
            "content": chat_reply,
        }
        conversation.append(reply_msg_obj)
        return conversation


class EmbeddingBasedChatContext(BaseChatContext):
    """Chat context."""

    def __init__(self, parent_chat):
        self.parent_chat = parent_chat
        self.context_file_path = GeneralConstants.EMBEDDINGS_FILE

    def add_user_input(self, conversation, user_input):
        user_input_msg_obj = {
            "role": "user",
            "name": self.parent_chat.username,
            "content": user_input,
        }
        store_message_to_file(
            msg_obj=user_input_msg_obj, file_path=self.context_file_path
        )
        intial_ai_instruct_msg = conversation[0]
        last_msg_exchange = conversation[-2:] if len(conversation) > 2 else []
        current_context = find_context(
            file_path=self.context_file_path,
            parent_chat=self.parent_chat,
            option="both",
        )
        conversation = [
            intial_ai_instruct_msg,
            *last_msg_exchange,
            *current_context,
            user_input_msg_obj,
        ]
        return conversation

    def add_chat_reply(self, conversation, chat_reply):
        reply_msg_obj = {
            "role": "assistant",
            "name": self.parent_chat.assistant_name,
            "content": chat_reply,
        }
        conversation.append(reply_msg_obj)
        store_message_to_file(file_path=self.context_file_path, msg_obj=reply_msg_obj)
        return conversation


class Chat:
    def __init__(
        self, model: str, base_instructions: str, send_full_history: bool = False
    ):
        self.model = model
        self.base_instructions = base_instructions
        self.send_full_history = send_full_history
        self.username = "chat_user"
        self.assistant_name = f"chat_{model.replace('.', '_')}"
        self.system_name = "chat_manager"

    def start(self):
        if self.send_full_history:
            context = BaseChatContext(parent_chat=self)
        else:
            context = EmbeddingBasedChatContext(parent_chat=self)

        TOTAL_N_TOKENS = 0
        initial_ai_instructions = (
            f"You are {self.assistant_name},"
            + f" a helpful assistant to {self.username}.\n"
            + "You answer correctly.\n"
            + f"{self.base_instructions.strip()}.\n"
            + f" Follow ALL directives by {self.system_name}."
        )

        conversation = [
            {
                "role": "system",
                "name": self.system_name,
                "content": initial_ai_instructions.strip(),
            }
        ]
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
                for token in communicate_with_model(
                    conversation=conversation, model=self.model
                ):
                    print(token, end="")
                    full_reply_content += token
                full_reply_content = full_reply_content.strip()
                print("\n")

                # Update context with the reply
                conversation = context.add_chat_reply(
                    conversation=conversation, chat_reply=full_reply_content
                )

                TOTAL_N_TOKENS += _num_tokens_from_string(
                    string="".join(msg["content"] for msg in conversation),
                    model=self.model,
                )
        except KeyboardInterrupt:
            print("Exiting.")
        print("TOTAL N TOKENS: ", TOTAL_N_TOKENS)


def communicate_with_model(conversation: list, model: str):
    success = False
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


def find_context(file_path: Path, parent_chat: Chat, option="both"):
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
        context_for_current_user_query += f"{msg['name']}: {msg['content']}\n"

    if not context_for_current_user_query:
        return []

    return [
        {
            "role": "system",
            "name": parent_chat.system_name,
            "content": f"{parent_chat.assistant_name}'s knowledge: "
            + f"{context_for_current_user_query} + Previous messages.\n"
            + f"Answer {parent_chat.username}'s last message.",
        }
    ]


def _num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))
