"""Chat context/history management."""

import ast
import itertools
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import openai
import pandas as pd
from scipy.spatial.distance import cosine as cosine_similarity

from .embeddings_database import EmbeddingsDatabase
from .general_utils import retry

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
        self._msg_fields_for_context = ["role", "content"]

    @property
    def embedding_model(self):
        """Return the embedding model used for context management."""
        return self.parent_chat.context_model

    @property
    def context_file_path(self):
        """Return the path to the context file."""
        return self.parent_chat.context_file_path

    def add_to_history(self, exchange_id: str, msg_list: list[dict]):
        """Add message exchange to history."""
        self.database.insert_message_exchange(
            exchange_id=exchange_id,
            chat_model=self.parent_chat.model,
            message_exchange=msg_list,
            embedding=self.request_embedding(msg_list=msg_list),
        )

    def load_history(self) -> list[dict]:
        """Load the chat history."""
        db_history_df = self.database.retrieve_history()

        # Convert unix timestamps to datetime objs at the local timezone
        db_history_df["timestamp"] = db_history_df["timestamp"].apply(
            lambda ts: datetime.fromtimestamp(ts)
            .replace(microsecond=0, tzinfo=timezone.utc)
            .astimezone(tz=None)
            .replace(tzinfo=None)
        )

        msg_exchanges = db_history_df["message_exchange"].apply(ast.literal_eval).tolist()
        # Add timestamps and path to eventual audio files to messages
        for i_msg_exchange, timestamp in enumerate(db_history_df["timestamp"]):
            # Index 0 is for the user's message, index 1 is for the assistant's reply
            msg_exchanges[i_msg_exchange][0]["timestamp"] = timestamp
            msg_exchanges[i_msg_exchange][1]["reply_audio_file_path"] = db_history_df[
                "reply_audio_file_path"
            ].iloc[i_msg_exchange]
            msg_exchanges[i_msg_exchange][1]["chat_model"] = db_history_df[
                "chat_model"
            ].iloc[i_msg_exchange]

        return list(itertools.chain.from_iterable(msg_exchanges))

    def get_context(self, msg: dict):
        """Return messages to serve as context for `msg` when requesting a completion."""
        return _make_list_of_context_msgs(
            history=self.select_relevant_history(msg=msg),
            system_name=self.parent_chat.system_name,
        )

    @abstractmethod
    def request_embedding(self, msg_list: list[dict]):
        """Request embedding from OpenAI API."""

    @abstractmethod
    def select_relevant_history(self, msg: dict):
        """Select chat history msgs to use as context for `msg`."""


class FullHistoryChatContext(ChatContext):
    """Context class using full chat history."""

    # Implement abstract methods
    def request_embedding(self, msg_list: list[dict]):  # noqa: ARG002
        """Return a placeholder embedding."""
        return

    def select_relevant_history(self, msg: dict):  # noqa: ARG002
        """Select chat history msgs to use as context for `msg`."""
        history = []
        for full_history_msg in self.load_history():
            history_msg = {
                k: v
                for k, v in full_history_msg.items()
                if k in self._msg_fields_for_context
            }
            history.append(history_msg)
        return history


class EmbeddingBasedChatContext(ChatContext):
    """Chat context using embedding models."""

    def request_embedding_for_text(self, text: str):
        """Request embedding for `text` from OpenAI according to used embedding model."""
        embedding_request = request_embedding_from_openai(
            text=text,
            model=self.embedding_model,
            openai_client=self.parent_chat.openai_client,
        )

        # Update parent chat's token usage db with tokens used in embedding request
        for db in [
            self.parent_chat.general_token_usage_db,
            self.parent_chat.token_usage_db,
        ]:
            for comm_type, n_tokens in embedding_request["tokens_usage"].items():
                input_or_output_kwargs = {f"n_{comm_type}_tokens": n_tokens}
                db.insert_data(model=self.embedding_model, **input_or_output_kwargs)

        return embedding_request["embedding"]

    # Implement abstract methods
    def request_embedding(self, msg_list: list[dict]):
        """Convert `msg_list` into a paragraph and get embedding from OpenAI API call."""
        text = "\n".join(
            [f"{msg['role'].strip()}: {msg['content'].strip()}" for msg in msg_list]
        )
        return self.request_embedding_for_text(text=text)

    def select_relevant_history(self, msg: dict):
        """Select chat history msgs to use as context for `msg`."""
        relevant_history = []
        for full_context_msg in _select_relevant_history(
            history_df=self.database.retrieve_history(),
            embedding=self.request_embedding_for_text(text=msg["content"]),
        ):
            context_msg = {
                k: v
                for k, v in full_context_msg.items()
                if k in self._msg_fields_for_context
            }
            relevant_history.append(context_msg)
        return relevant_history


@retry()
def request_embedding_from_openai(text: str, model: str, openai_client: openai.OpenAI):
    """Request embedding for `text` according to context model `model` from OpenAI."""
    text = text.strip()
    embedding_request = openai_client.embeddings.create(input=[text], model=model)

    embedding = embedding_request.data[0].embedding

    input_tokens = embedding_request.usage.prompt_tokens
    output_tokens = embedding_request.usage.total_tokens - input_tokens
    tokens_usage = {"input": input_tokens, "output": output_tokens}

    return {"embedding": embedding, "tokens_usage": tokens_usage}


def _make_list_of_context_msgs(history: list[dict], system_name: str):
    sys_directives = "Considering the previous messages, answer the next message:"
    sys_msg = {"role": "system", "name": system_name, "content": sys_directives}
    return [*history, sys_msg]


def _select_relevant_history(
    history_df: pd.DataFrame,
    embedding: np.ndarray,
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
