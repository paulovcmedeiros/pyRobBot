import ast
import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity

from . import GeneralConstants

if TYPE_CHECKING:
    from .chat import Chat


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
        tokens_usage = {"input": 0, "output": 0}

        return {"conversation": conversation, "tokens_usage": tokens_usage}

    def add_chat_reply(self, conversation: list, chat_reply: str):
        reply_msg_obj = {
            "role": "assistant",
            "name": self.parent_chat.assistant_name,
            "content": chat_reply,
        }
        conversation.append(reply_msg_obj)
        tokens_usage = {"input": 0, "output": 0}

        return {"conversation": conversation, "tokens_usage": tokens_usage}


class EmbeddingBasedChatContext(BaseChatContext):
    """Chat context."""

    def __init__(self, embedding_model: str, parent_chat: "Chat"):
        self.embedding_model = embedding_model
        self.parent_chat = parent_chat
        self.context_file_path = GeneralConstants.EMBEDDINGS_FILE

    def calculate_embedding(self, text: str):
        text = text.lower().replace("\n", " ")
        return request_embedding_from_openai(text=text, model=self.embedding_model)

    def add_to_history(self, text, embedding):
        _store_message_and_embedding(
            file_path=self.context_file_path, msg_obj=text, embedding=embedding
        )

    def get_context(self, embedding):
        return _find_context(
            embedding=embedding,
            file_path=self.context_file_path,
            parent_chat=self.parent_chat,
        )


def request_embedding_from_openai(text: str, model: str):
    text = text.replace("\n", " ")
    embedding_request = openai.Embedding.create(input=[text], model=model)

    embedding = embedding_request["data"][0]["embedding"]

    input_tokens = embedding_request["usage"]["prompt_tokens"]
    output_tokens = embedding_request["usage"]["total_tokens"] - input_tokens
    tokens_usage = {"input": input_tokens, "output": output_tokens}

    return {"embedding": embedding, "tokens_usage": tokens_usage}


def _store_message_and_embedding(
    msg_obj: dict, embedding, file_path: Path = GeneralConstants.EMBEDDINGS_FILE
):
    """Store message and embeddings to file."""
    # Adapted from <https://community.openai.com/t/
    #  use-embeddings-to-retrieve-relevant-context-for-ai-assistant/268538>
    # See also <https://platform.openai.com/docs/guides/embeddings>.

    emb_mess_pair = {
        "timestamp": int(time.time()),
        "message": json.dumps(msg_obj),
        "embedding": json.dumps(embedding),
    }

    init_file = not file_path.exists() or file_path.stat().st_size == 0
    write_mode = "w" if init_file else "a"

    with open(file_path, write_mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=emb_mess_pair.keys())
        if init_file:
            writer.writeheader()
        writer.writerow(emb_mess_pair)


def _find_context(
    file_path: Path,
    embedding: list[float],
    parent_chat: "Chat",
    n_related_entries: int = 4,
    n_directly_preceeding_exchanges: int = 2,
):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return []

    df["embedding"] = df.embedding.apply(ast.literal_eval).apply(np.array)

    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, embedding))

    # Get the last n messages added to the history
    df_last_n_chats = df.tail(n_directly_preceeding_exchanges)

    df_similar_chats = (
        df.sort_values("similarity", ascending=False)
        .head(n_related_entries)
        .sort_values("timestamp")
    )
    df_context = pd.concat([df_similar_chats, df_last_n_chats])
    selected = df_context["message"].apply(ast.literal_eval).drop_duplicates().tolist()

    context_msg_content = "You know previous messages.\n"
    context_msg_content = "You also know that the following was said:\n"
    for message in selected:
        context_msg_content += f"{message}\n"
    context_msg_content += "Answer the last message."
    context_msg = {
        "role": "system",
        "name": parent_chat.system_name,
        "content": context_msg_content,
    }

    return [context_msg]
