import ast
import csv
import itertools
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

    def add_to_history(self, msg_list: list[dict]):
        self.history += msg_list
        return self._tokens_usage

    def load_history(self):
        """Load the chat history."""
        return self.history

    def get_context(self, msg: dict):
        context_msgs = _make_list_of_context_msgs(
            history=self.history, system_name=self.parent_chat.system_name
        )
        return {"context_messages": context_msgs, "tokens_usage": self._tokens_usage}


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

    def add_to_history(self, msg_list: list[dict]):
        embedding_request = self._calculate_embedding_for_msgs(msg_list=msg_list)
        _store_message_exchange_and_corresponding_embedding(
            msg_list=msg_list,
            embedding_model=self.embedding_model,
            chat_model=self.parent_chat.model,
            embedding=embedding_request["embedding"],
            file_path=self.context_file_path,
        )
        return embedding_request["tokens_usage"]

    def load_history(self):
        """Load the chat history from file."""
        try:
            df = pd.read_csv(self.context_file_path)
        except FileNotFoundError:
            return []
        selected_history = (df["message_exchange"].apply(ast.literal_eval)).tolist()
        selected_history = list(itertools.chain.from_iterable(selected_history))
        return selected_history

    def get_context(self, msg: dict):
        embedding_request = self._calculate_embedding_for_text(text=msg["content"])
        context_messages = _find_context(
            embedding=embedding_request["embedding"],
            file_path=self.context_file_path,
            parent_chat=self.parent_chat,
        )

        return {
            "context_messages": context_messages,
            "tokens_usage": embedding_request["tokens_usage"],
        }

    def _calculate_embedding_for_msgs(self, msg_list: list[dict]):
        text = "\n".join(
            [f"{msg['role'].strip()}: {msg['content'].strip()}" for msg in msg_list]
        )
        return self._calculate_embedding_for_text(text=text)

    def _calculate_embedding_for_text(self, text: str):
        return request_embedding_from_openai(text=text, model=self.embedding_model)


@retry_api_call()
def request_embedding_from_openai(text: str, model: str):
    text = text.strip()
    embedding_request = openai.Embedding.create(input=[text], model=model)

    embedding = embedding_request["data"][0]["embedding"]

    input_tokens = embedding_request["usage"]["prompt_tokens"]
    output_tokens = embedding_request["usage"]["total_tokens"] - input_tokens
    tokens_usage = {"input": input_tokens, "output": output_tokens}

    return {"embedding": embedding, "tokens_usage": tokens_usage}


def _store_message_exchange_and_corresponding_embedding(
    msg_list: list[dict],
    embedding_model: str,
    chat_model: str,
    embedding: list[float],
    file_path: Path,
):
    """Store message and embeddings to file."""
    # Adapted from <https://community.openai.com/t/
    #  use-embeddings-to-retrieve-relevant-context-for-ai-assistant/268538>
    # See also <https://platform.openai.com/docs/guides/embeddings>.
    embedding_file_entry_data = {
        "timestamp": int(time.time()),
        "embedding_model": f"{embedding_model}",
        "chat_model": f"{chat_model}",
        "message_exchange": json.dumps(msg_list),
        "embedding": json.dumps(embedding),
    }

    init_file = not file_path.exists() or file_path.stat().st_size == 0
    write_mode = "w" if init_file else "a"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, write_mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=embedding_file_entry_data.keys())
        if init_file:
            writer.writeheader()
        writer.writerow(embedding_file_entry_data)


def _make_list_of_context_msgs(history: list[dict], system_name: str):
    sys_directives = "Considering the previous messages, answer the next message:"
    sys_msg = {"role": "system", "name": system_name, "content": sys_directives}
    return [*history, sys_msg]


def _find_context(
    file_path: Path,
    embedding: list[float],
    parent_chat: "Chat",
    n_related_msg_exchanges: int = 3,
    n_tailing_history_exchanges: int = 2,
):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return []

    df = df.loc[df["embedding_model"] == parent_chat.context_model]
    df["embedding"] = df["embedding"].apply(ast.literal_eval).apply(np.array)

    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, embedding))

    # Get the last messages added to the history
    df_last_n_chats = df.tail(n_tailing_history_exchanges)

    # Get the most similar messages
    df_similar_chats = (
        df.sort_values("similarity", ascending=False)
        .head(n_related_msg_exchanges)
        .sort_values("timestamp")
    )

    df_context = pd.concat([df_similar_chats, df_last_n_chats])
    selected_history = (
        df_context["message_exchange"].apply(ast.literal_eval).drop_duplicates()
    ).tolist()

    selected_history = list(itertools.chain.from_iterable(selected_history))

    return _make_list_of_context_msgs(
        history=selected_history, system_name=parent_chat.system_name
    )
