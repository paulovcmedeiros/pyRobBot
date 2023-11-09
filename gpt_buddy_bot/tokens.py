import datetime
import sqlite3
from pathlib import Path

import pandas as pd
import tiktoken

# See <https://openai.com/pricing> for the latest prices.
PRICE_PER_K_TOKENS = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-3.5-turbo-16k": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
    "gpt-4-1106-preview": {"input": 0.03, "output": 0.06},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
    "full-history": {"input": 0.0, "output": 0.0},
}


class TokenUsageDatabase:
    def __init__(self, fpath: Path):
        self.fpath = fpath
        self.token_price = {}
        for model, price_per_k_tokens in PRICE_PER_K_TOKENS.items():
            self.token_price[model] = {
                k: v / 1000.0 for k, v in price_per_k_tokens.items()
            }

        self.create()

    def create(self):
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.fpath)
        cursor = conn.cursor()

        # Create a table to store the data with 'timestamp' as the primary key
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS token_costs (
                timestamp REAL PRIMARY KEY,
                model TEXT,
                n_input_tokens INTEGER,
                n_output_tokens INTEGER,
                cost_input_tokens REAL,
                cost_output_tokens REAL
            )
        """
        )

        conn.commit()
        conn.close()

    # Function to insert data into the database
    def insert_data(self, model, n_input_tokens, n_output_tokens):
        if model is None:
            return

        conn = sqlite3.connect(self.fpath)
        cursor = conn.cursor()

        # Insert the data into the table
        cursor.execute(
            """
        INSERT OR REPLACE INTO token_costs (
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
                datetime.datetime.utcnow().timestamp(),
                model,
                n_input_tokens,
                n_output_tokens,
                n_input_tokens * self.token_price[model]["input"],
                n_output_tokens * self.token_price[model]["output"],
            ),
        )

        conn.commit()
        conn.close()

    def retrieve_sums_by_model(self):
        conn = sqlite3.connect(self.fpath)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                model,
                MIN(timestamp) AS earliest_timestamp,
                SUM(n_input_tokens) AS total_n_input_tokens,
                SUM(n_output_tokens) AS total_n_output_tokens,
                SUM(cost_input_tokens) AS total_cost_input_tokens,
                SUM(cost_output_tokens) AS total_cost_output_tokens
            FROM token_costs
            GROUP BY model
            """
        )

        data = cursor.fetchall()

        conn.close()

        sums_by_model = {}
        for row in data:
            model_name = row[0]
            sums = {
                "earliest_timestamp": row[1],
                "n_input_tokens": row[2],
                "n_output_tokens": row[3],
                "cost_input_tokens": row[4],
                "cost_output_tokens": row[5],
            }
            sums_by_model[model_name] = sums

        return sums_by_model

    def get_usage_balance_dataframe(self):
        sums_by_model = self.retrieve_sums_by_model()
        df_rows = []
        for model, accumulated_usage in sums_by_model.items():
            if model is None:
                continue

            accumulated_tokens_usage = {
                "input": accumulated_usage["n_input_tokens"],
                "output": accumulated_usage["n_output_tokens"],
            }
            accumlated_costs = {
                "input": accumulated_usage["cost_input_tokens"],
                "output": accumulated_usage["cost_output_tokens"],
            }
            first_used = datetime.datetime.fromtimestamp(
                accumulated_usage["earliest_timestamp"], datetime.timezone.utc
            ).isoformat(sep=" ", timespec="seconds")
            df_row = {
                "Model": model,
                "First Registered Use": first_used.replace("+00:00", "Z"),
                "Tokens: Input": accumulated_tokens_usage["input"],
                "Tokens: Output": accumulated_tokens_usage["output"],
                "Tokens: Total": sum(accumulated_tokens_usage.values()),
                "Cost ($): Input": accumlated_costs["input"],
                "Cost ($): Output": accumlated_costs["output"],
                "Cost ($): Total": sum(accumlated_costs.values()),
            }
            df_rows.append(df_row)

        df = pd.DataFrame(df_rows)
        if not df.empty:
            df = _add_totals_row(_group_columns_by_prefix(df))

        return df

    def get_current_chat_usage_dataframe(self, token_usage_per_model: dict):
        df_rows = []
        for model, token_usage in token_usage_per_model.items():
            if model is None:
                continue

            costs = {k: v * self.token_price[model][k] for k, v in token_usage.items()}
            df_row = {
                "Model": model,
                "Tokens: Input": token_usage["input"],
                "Tokens: Output": token_usage["output"],
                "Tokens: Total": sum(token_usage.values()),
                "Cost ($): Input": costs["input"],
                "Cost ($): Output": costs["output"],
                "Cost ($): Total": sum(costs.values()),
            }
            df_rows.append(df_row)
        df = pd.DataFrame(df_rows)
        if df_rows:
            df = _group_columns_by_prefix(df.set_index("Model"))
            df = _add_totals_row(df)
        return df

    def print_usage_costs(self, token_usage: dict, current_chat: bool = True):
        header_start = "Estimated token usage and associated costs"
        header2dataframe = {
            f"{header_start}: Accumulated": self.get_usage_balance_dataframe(),
            f"{header_start}: Current Chat": self.get_current_chat_usage_dataframe(
                token_usage
            ),
        }

        for header, df in header2dataframe.items():
            if "current" in header.lower() and not current_chat:
                continue
            _print_df(df=df, header=header)

        print()
        print("Note: These are only estimates. Actual costs may vary.")
        link = "https://platform.openai.com/account/usage"
        print(f"Please visit <{link}> to follow your actual usage and costs.")


def get_n_tokens_from_msgs(messages: list[dict], model: str):
    """Returns the number of tokens used by a list of messages."""
    # Adapted from
    # <https://platform.openai.com/docs/guides/text-generation/managing-tokens>
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # OpenAI's original function was implemented for gpt-3.5-turbo-0613, but we'll use
    # it for all models for now. We are only intereste dinestimates, after all.
    num_tokens = 0
    for message in messages:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def _group_columns_by_prefix(df):
    df = df.copy()
    col_tuples_for_multiindex = df.columns.str.split(": ", expand=True).values
    df.columns = pd.MultiIndex.from_tuples(
        [("", x[0]) if pd.isnull(x[1]) else x for x in col_tuples_for_multiindex]
    )
    return df


def _add_totals_row(df):
    df = df.copy()
    dtypes = df.dtypes
    df.loc["Total"] = df.sum(numeric_only=True)
    for col in df.columns:
        df[col] = df[col].astype(dtypes[col])
    df = df.fillna("")
    return df


def _print_df(df: pd.DataFrame, header: str):
    underline = "-" * len(header)
    print()
    print(underline)
    print(header)
    print(underline)
    if df.empty or df.loc["Total"]["Tokens"]["Total"] == 0:
        print("None.")
    else:
        print(df)
    print()
