import datetime
import sqlite3
from pathlib import Path

PRICING_PER_THOUSAND_TOKENS = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
}


class TokenUsageDatabase:
    def __init__(self, fpath: Path, model: str):
        self.fpath = fpath
        self.token_price = {
            k: v / 1000.0 for k, v in PRICING_PER_THOUSAND_TOKENS[model].items()
        }

    # Function to create the database and table
    def create(self):
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.fpath)
        cursor = conn.cursor()

        # Create a table to store the data with 'timestamp' as the primary key
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS token_costs (
                timestamp REAL PRIMARY KEY,
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
        INSERT OR REPLACE INTO token_costs
        (timestamp, n_input_tokens, n_output_tokens, cost_input_tokens, cost_output_tokens)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                datetime.datetime.utcnow().timestamp(),
                n_input_tokens,
                n_output_tokens,
                n_input_tokens * self.token_price["input"],
                n_output_tokens * self.token_price["output"],
            ),
        )

        conn.commit()
        conn.close()

    def retrieve_sums(self):
        conn = sqlite3.connect(self.fpath)
        cursor = conn.cursor()

        # SQL query to calculate the sum of each variable
        cursor.execute(
            """
            SELECT
                MIN(timestamp) AS earliest_timestamp,
                SUM(n_input_tokens) AS total_n_input_tokens,
                SUM(n_output_tokens) AS total_n_output_tokens,
                SUM(cost_input_tokens) AS total_cost_input_tokens,
                SUM(cost_output_tokens) AS total_cost_output_tokens
            FROM token_costs
        """
        )

        data = cursor.fetchone()

        conn.close()

        return {
            "earliest_timestamp": data[0],
            "n_input_tokens": data[1],
            "n_output_tokens": data[2],
            "cost_input_tokens": data[3],
            "cost_output_tokens": data[4],
        }
