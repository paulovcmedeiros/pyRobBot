import datetime
import sqlite3
from collections import defaultdict
from pathlib import Path

import tiktoken

PRICING_PER_THOUSAND_TOKENS = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
}


class TokenUsageDatabase:
    def __init__(self, fpath: Path, model: str):
        self.fpath = fpath
        self.model = model.strip()
        pricing = PRICING_PER_THOUSAND_TOKENS[self.model]
        self.token_price = {k: v / 1000.0 for k, v in pricing.items()}
        self.create()

    def get_n_tokens(self, string: str) -> int:
        return _num_tokens_from_string(string=string, model=self.model)

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
    def insert_data(self, n_input_tokens, n_output_tokens):
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
                self.model,
                n_input_tokens,
                n_output_tokens,
                n_input_tokens * self.token_price["input"],
                n_output_tokens * self.token_price["output"],
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

    def retrieve_sums(self):
        sums = defaultdict(int)
        for sums_by_model in self.retrieve_sums_by_model().values():
            for k, v in sums_by_model.items():
                sums[k] += v
        return sums

    def print_usage_costs(self, token_usage: dict):
        print()
        print("=======================================================")
        print("Summary of OpenAI API token usage and associated costs:")
        print("=======================================================")

        for model, accumulated_usage in self.retrieve_sums_by_model().items():
            _print_accumulated_token_usage(
                accumulated_usage=accumulated_usage, model=model
            )

        _print_accumulated_token_usage(accumulated_usage=self.retrieve_sums())

        print()
        print("Token usage summary for this chat:")
        for k, v in token_usage.items():
            print(f"    > {k.capitalize()}: {v}")
        print(f"    > Total:  {sum(token_usage.values())}")
        costs = {k: v * self.token_price[k] for k, v in token_usage.items()}
        print(f"Estimated total cost for this chat: ${sum(costs.values()):.3f}.")


def _num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


def _print_accumulated_token_usage(accumulated_usage: dict, model: str = None):
    print()
    if model is not None:
        print(f"Model: {model}")

    since = datetime.datetime.fromtimestamp(
        accumulated_usage["earliest_timestamp"], datetime.timezone.utc
    ).isoformat(sep=" ", timespec="seconds")
    print(f"Accumulated token usage since {since.replace('+00:00', 'Z')}:")

    accumulated_token_usage = {
        "input": accumulated_usage["n_input_tokens"],
        "output": accumulated_usage["n_output_tokens"],
    }
    acc_costs = {
        "input": accumulated_usage["cost_input_tokens"],
        "output": accumulated_usage["cost_output_tokens"],
    }
    for k, v in accumulated_token_usage.items():
        print(f"    > {k.capitalize()}: {v}")
    print(f"    > Total:  {sum(accumulated_token_usage.values())}")
    print(f"Estimated total costs since same date: ${sum(acc_costs.values()):.3f}.")
