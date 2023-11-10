"""Chat context/history management."""
import ast
import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity

from .embeddings_database import EmbeddingsDatabase
from .openai_utils import retry_api_call

if TYPE_CHECKING:
    from .chat import Chat


class ChatContext(ABC):
    """Abstract base class for representing the context of a chat."""

    def __init__(self, parent_chat: "Chat"):
        """Initialise the instance given a parent `Chat` object."""
        self.parent_chat = parent_chat
        self.database = EmbeddingsDatabase(
            db_path=self.context_file_path, embedding_model=self.embedding_model
        )

    @property
    def embedding_model(self):
        """Return the embedding model used for context management."""
        return self.parent_chat.context_model

    @property
    def context_file_path(self):
        """Return the path to the context file."""
        return self.parent_chat.context_file_path

    def add_to_history(self, msg_list: list[dict]):
        """Add message exchange to history."""
        embedding_request = self.request_embedding(msg_list=msg_list)
        self.database.insert_message_exchange(
            chat_model=self.parent_chat.model,
            message_exchange=msg_list,
            embedding=embedding_request["embedding"],
        )
        return embedding_request["tokens_usage"]

    def load_history(self) -> list[dict]:
        """Load the chat history."""
        messages_df = self.database.get_messages_dataframe()
        msg_exchanges = messages_df["message_exchange"].apply(ast.literal_eval).tolist()
        return list(itertools.chain.from_iterable(msg_exchanges))

    @abstractmethod
    def request_embedding(self, msg_list: list[dict]):
        """Request embedding from OpenAI API."""

    @abstractmethod
    def get_context(self, msg: dict):
        """Return context messages."""


class FullHistoryChatContext(ChatContext):
    """Context class using full chat history."""

    def __init__(self, *args, **kwargs):
        """Initialise instance. Args and kwargs are passed to the parent class' `init`."""
        super().__init__(*args, **kwargs)
        self._placeholder_tokens_usage = {"input": 0, "output": 0}

    # Implement abstract methods
    def request_embedding(self, msg_list: list[dict]):  # noqa: ARG002
        """Return a placeholder embedding request."""
        return {"embedding": None, "tokens_usage": self._placeholder_tokens_usage}

    def get_context(self, msg: dict):  # noqa: ARG002
        """Return context messages."""
        context_msgs = _make_list_of_context_msgs(
            history=self.load_history(), system_name=self.parent_chat.system_name
        )
        return {
            "context_messages": context_msgs,
            "tokens_usage": self._placeholder_tokens_usage,
        }


class EmbeddingBasedChatContext(ChatContext):
    """Chat context using embedding models."""

    def _request_embedding_for_text(self, text: str):
        return request_embedding_from_openai(text=text, model=self.embedding_model)

    # Implement abstract methods
    def request_embedding(self, msg_list: list[dict]):
        """Request embedding from OpenAI API."""
        text = "\n".join(
            [f"{msg['role'].strip()}: {msg['content'].strip()}" for msg in msg_list]
        )
        return self._request_embedding_for_text(text=text)

    def get_context(self, msg: dict):
        """Return context messages."""
        embedding_request = self._request_embedding_for_text(text=msg["content"])
        selected_history = _select_relevant_history(
            history_df=self.database.get_messages_dataframe(),
            embedding=embedding_request["embedding"],
        )
        context_messages = _make_list_of_context_msgs(
            history=selected_history, system_name=self.parent_chat.system_name
        )
        return {
            "context_messages": context_messages,
            "tokens_usage": embedding_request["tokens_usage"],
        }


@retry_api_call()
def request_embedding_from_openai(text: str, model: str):
    """Request embedding for `text` according to context model `model` from OpenAI."""
    text = text.strip()
    embedding_request = openai.Embedding.create(input=[text], model=model)

    embedding = embedding_request["data"][0]["embedding"]

    input_tokens = embedding_request["usage"]["prompt_tokens"]
    output_tokens = embedding_request["usage"]["total_tokens"] - input_tokens
    tokens_usage = {"input": input_tokens, "output": output_tokens}

    return {"embedding": embedding, "tokens_usage": tokens_usage}


def _make_list_of_context_msgs(history: list[dict], system_name: str):
    sys_directives = "Considering the previous messages, answer the next message:"
    sys_msg = {"role": "system", "name": system_name, "content": sys_directives}
    return [*history, sys_msg]


def _select_relevant_history(
    history_df: pd.DataFrame,
    embedding: list[float],
    max_n_prompt_reply_pairs: int = 5,
    max_n_tailing_prompt_reply_pairs: int = 2,
):
    history_df["embedding"] = (
        history_df["embedding"].apply(ast.literal_eval).apply(np.array)
    )
    history_df["similarity"] = history_df["embedding"].apply(
        lambda x: cosine_similarity(x, embedding)
    )

    # Get the last messages added to the history
    df_last_n_chats = history_df.tail(max_n_tailing_prompt_reply_pairs)

    # Get the most similar messages
    df_similar_chats = (
        history_df.sort_values("similarity", ascending=False)
        .head(max_n_prompt_reply_pairs)
        .sort_values("timestamp")
    )

    df_context = pd.concat([df_similar_chats, df_last_n_chats])
    selected_history = (
        df_context["message_exchange"].apply(ast.literal_eval).drop_duplicates()
    ).tolist()

    return list(itertools.chain.from_iterable(selected_history))
