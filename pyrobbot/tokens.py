"""Management of token usage and costs for OpenAI API."""

import contextlib
import datetime
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import tiktoken

# See <https://openai.com/pricing> for the latest prices.
PRICE_PER_K_TOKENS_LLM = {
    # Continuous model upgrades (models that point to the latest versions)
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo-16k": {"input": 0.001, "output": 0.002},  # -> gpt-3.5-turbo-16k-0613
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    # Static model versions
    # GPT 3
    "gpt-3.5-turbo-0125": {"input": 0.0015, "output": 0.002},
    "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002},  # Deprecated, 2024-06-13
    "gpt-3.5-turbo-16k-0613": {"input": 0.001, "output": 0.002},  # Deprecated, 2024-06-13
    # GPT 4
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-0613": {"input": 0.03, "output": 0.06},
    "gpt-4-32k-0613": {"input": 0.06, "output": 0.12},
}
PRICE_PER_K_TOKENS_EMBEDDINGS = {
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
    "text-embedding-ada-002-v2": {"input": 0.0001, "output": 0.0},
    "text-davinci:002": {"input": 0.0020, "output": 0.020},
    "full-history": {"input": 0.0, "output": 0.0},
}
PRICE_PER_K_TOKENS_TTS_AND_STT = {
    "tts-1": {"input": 0.015, "output": 0.0},
    "tts-1-hd": {"input": 0.03, "output": 0.0},
    "whisper-1": {"input": 0.006, "output": 0.0},
}
PRICE_PER_K_TOKENS = (
    PRICE_PER_K_TOKENS_LLM
    | PRICE_PER_K_TOKENS_EMBEDDINGS
    | PRICE_PER_K_TOKENS_TTS_AND_STT
)


class TokenUsageDatabase:
    """Manages a database to store estimated token usage and costs for OpenAI API."""

    def __init__(self, fpath: Path):
        """Initialize a TokenUsageDatabase instance."""
        self.fpath = fpath
        self.token_price = {}
        for model, price_per_k_tokens in PRICE_PER_K_TOKENS.items():
            self.token_price[model] = {
                k: v / 1000.0 for k, v in price_per_k_tokens.items()
            }

        self.create()

    def create(self):
        """Create the database if it doesn't exist."""
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.fpath)
        cursor = conn.cursor()

        # Create a table to store the data with 'timestamp' as the primary key
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS token_costs (
                timestamp INTEGER NOT NULL,
                model TEXT NOT NULL,
                n_input_tokens INTEGER NOT NULL,
                n_output_tokens INTEGER NOT NULL,
                cost_input_tokens REAL NOT NULL,
                cost_output_tokens REAL NOT NULL
            )
        """
        )

        conn.commit()
        conn.close()

    def insert_data(
        self,
        model: str,
        n_input_tokens: int = 0,
        n_output_tokens: int = 0,
        timestamp: Optional[int] = None,
    ):
        """Insert the data into the token_costs table."""
        if model is None:
            return

        conn = sqlite3.connect(self.fpath)
        cursor = conn.cursor()

        # Insert the data into the table
        cursor.execute(
            """
        INSERT INTO token_costs (
            timestamp,
            model,
            n_input_tokens,
            n_output_tokens,
            cost_input_tokens,
            cost_output_tokens
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                timestamp or int(datetime.datetime.utcnow().timestamp()),
                model,
                n_input_tokens,
                n_output_tokens,
                n_input_tokens * self.token_price[model]["input"],
                n_output_tokens * self.token_price[model]["output"],
            ),
        )

        conn.commit()
        conn.close()

    def get_usage_balance_dataframe(self):
        """Get a dataframe with the accumulated token usage and costs."""
        conn = sqlite3.connect(self.fpath)
        query = """
            SELECT
                model as Model,
                MIN(timestamp) AS "First Used",
                SUM(n_input_tokens) AS "Tokens: In",
                SUM(n_output_tokens) AS "Tokens: Out",
                SUM(n_input_tokens + n_output_tokens) AS "Tokens: Tot.",
                SUM(cost_input_tokens) AS "Cost ($): In",
                SUM(cost_output_tokens) AS "Cost ($): Out",
                SUM(cost_input_tokens + cost_output_tokens) AS "Cost ($): Tot."
            FROM token_costs
            GROUP BY model
            ORDER BY "Cost ($): Tot." DESC
        """

        usage_df = pd.read_sql_query(query, con=conn)
        conn.close()

        usage_df["First Used"] = pd.to_datetime(usage_df["First Used"], unit="s")

        usage_df = _group_columns_by_prefix(_add_totals_row(usage_df))

        # Add metadata to returned dataframe
        usage_df.attrs["description"] = "Estimated token usage and associated costs"
        link = "https://platform.openai.com/account/usage"
        disclaimers = [
            "Note: These are only estimates. Actual costs may vary.",
            f"Please visit <{link}> to follow your actual usage and costs.",
        ]
        usage_df.attrs["disclaimer"] = "\n".join(disclaimers)

        return usage_df


def get_n_tokens_from_msgs(messages: list[dict], model: str):
    """Returns the number of tokens used by a list of messages."""
    # Adapted from
    # <https://platform.openai.com/docs/guides/text-generation/managing-tokens>
    encoding = tiktoken.get_encoding("cl100k_base")
    with contextlib.suppress(KeyError):
        encoding = tiktoken.encoding_for_model(model)

    # OpenAI's original function was implemented for gpt-3.5-turbo-0613, but we'll use
    # it for all models for now. We are only interested in estimates, after all.
    num_tokens = 0
    for message in messages:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            if not isinstance(value, str):
                raise TypeError(
                    f"Value for key '{key}' has type {type(value)}. Expected str: {value}"
                )
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def _group_columns_by_prefix(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    col_tuples_for_multiindex = dataframe.columns.str.split(": ", expand=True).to_numpy()
    dataframe.columns = pd.MultiIndex.from_tuples(
        [("", x[0]) if pd.isna(x[1]) else x for x in col_tuples_for_multiindex]
    )
    return dataframe


def _add_totals_row(accounting_df: pd.DataFrame):
    dtypes = accounting_df.dtypes
    sums_df = accounting_df.sum(numeric_only=True).rename("Total").to_frame().T
    return pd.concat([accounting_df, sums_df]).astype(dtypes).fillna(" ")
