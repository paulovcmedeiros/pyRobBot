import ast
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings

from . import GeneralConstants

if TYPE_CHECKING:
    from .chat_gpt import Chat


class BaseChatContext:
    def __init__(self, parent_chat: "Chat"):
        self.parent_chat = parent_chat

    def add_user_input(self, conversation: list, user_input: str):
        user_input_msg_obj = {
            "role": "user",
            "name": self.parent_chat.username,
            "content": user_input,
        }
        conversation.append(user_input_msg_obj)
        return conversation

    def add_chat_reply(self, conversation: list, chat_reply: str):
        reply_msg_obj = {
            "role": "assistant",
            "name": self.parent_chat.assistant_name,
            "content": chat_reply,
        }
        conversation.append(reply_msg_obj)
        return conversation


class EmbeddingBasedChatContext(BaseChatContext):
    """Chat context."""

    def __init__(self, parent_chat: "Chat"):
        self.parent_chat = parent_chat
        self.context_file_path = GeneralConstants.EMBEDDINGS_FILE

    def add_user_input(self, conversation: list, user_input: str):
        user_input_msg_obj = {
            "role": "user",
            "name": self.parent_chat.username,
            "content": user_input,
        }
        _store_message_to_file(
            msg_obj=user_input_msg_obj, file_path=self.context_file_path
        )
        intial_ai_instruct_msg = conversation[0]
        last_msg_exchange = conversation[-2:] if len(conversation) > 2 else []
        current_context = _find_context(
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

    def add_chat_reply(self, conversation: list, chat_reply: str):
        reply_msg_obj = {
            "role": "assistant",
            "name": self.parent_chat.assistant_name,
            "content": chat_reply,
        }
        conversation.append(reply_msg_obj)
        _store_message_to_file(file_path=self.context_file_path, msg_obj=reply_msg_obj)
        return conversation


def _store_message_to_file(
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


def _find_context(file_path: Path, parent_chat: "Chat", option="both"):
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
            + "Only answer last message.",
        }
    ]
