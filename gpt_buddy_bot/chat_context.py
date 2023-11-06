import ast
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity

if TYPE_CHECKING:
    from .chat import Chat


class BaseChatContext:
    def __init__(self, parent_chat: "Chat"):
        self.parent_chat = parent_chat
        self.history = deque(maxlen=50)
        self._tokens_usage = {"input": 0, "output": 0}

    def add_to_history(self, text: str):
        self.history.append(text)
        return self._tokens_usage

    def get_context(self, text: str):
        context_msg = _compose_context_msg(
            history=self.history, system_name=self.parent_chat.system_name
        )
        return {"context_messages": [context_msg], "tokens_usage": self._tokens_usage}


class EmbeddingBasedChatContext(BaseChatContext):
    """Chat context."""

    def __init__(self, parent_chat: "Chat"):
        self.parent_chat = parent_chat

    @property
    def embedding_model(self):
        return self.parent_chat.context_model

    @property
    def context_file_path(self):
        return self.parent_chat.context_file_path

    def add_to_history(self, text: str):
        embedding_request = self.calculate_embedding(text=text)
        _store_message_embedding_data(
            obj=text,
            embedding_model=self.embedding_model,
            chat_model=self.parent_chat.model,
            embedding=embedding_request["embedding"],
            file_path=self.context_file_path,
        )
        return embedding_request["tokens_usage"]

    def get_context(self, text: str):
        embedding_request = self.calculate_embedding(text=text)
        context_messages = _find_context(
            embedding=embedding_request["embedding"],
            file_path=self.context_file_path,
            parent_chat=self.parent_chat,
        )

        return {
            "context_messages": context_messages,
            "tokens_usage": embedding_request["tokens_usage"],
        }

    def calculate_embedding(self, text: str):
        return request_embedding_from_openai(text=text, model=self.embedding_model)


def request_embedding_from_openai(text: str, model: str):
    text.lower().replace("\n", " ")
    embedding_request = openai.Embedding.create(input=[text], model=model)

    embedding = embedding_request["data"][0]["embedding"]

    input_tokens = embedding_request["usage"]["prompt_tokens"]
    output_tokens = embedding_request["usage"]["total_tokens"] - input_tokens
    tokens_usage = {"input": input_tokens, "output": output_tokens}

    return {"embedding": embedding, "tokens_usage": tokens_usage}


def _store_message_embedding_data(
    obj, embedding_model: str, chat_model: str, embedding: list[float], file_path: Path
):
    """Store message and embeddings to file."""
    # Adapted from <https://community.openai.com/t/
    #  use-embeddings-to-retrieve-relevant-context-for-ai-assistant/268538>
    # See also <https://platform.openai.com/docs/guides/embeddings>.

    embedding_file_entry_data = {
        "timestamp": int(time.time()),
        "embedding_model": f"{embedding_model}",
        "chat_model": f"{chat_model}",
        "message": json.dumps(obj),
        "embedding": json.dumps(embedding),
    }

    init_file = not file_path.exists() or file_path.stat().st_size == 0
    write_mode = "w" if init_file else "a"

    with open(file_path, write_mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=embedding_file_entry_data.keys())
        if init_file:
            writer.writeheader()
        writer.writerow(embedding_file_entry_data)


def _compose_context_msg(history: list[str], system_name: str):
    context_msg_content = "You know that the following was said:\n\n"
    context_msg_content += "\x1f\n".join(rf"{message}" for message in history) + "\n\n"
    context_msg_content += "Answer the last message."
    return {"role": "system", "name": system_name, "content": context_msg_content}


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

    df = df.loc[df["embedding_model"] == parent_chat.context_model]
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

    return [_compose_context_msg(history=selected, system_name=parent_chat.system_name)]
