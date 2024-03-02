"""Management of embeddings/chat history storage and retrieval."""

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger


class EmbeddingsDatabase:
    """Class for managing an SQLite database storing embeddings and associated data."""

    def __init__(self, db_path: Path, embedding_model: str):
        """Initialise the EmbeddingsDatabase object.

        Args:
            db_path (Path): The path to the SQLite database file.
            embedding_model (str): The embedding model associated with this database.
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.create()

    def create(self):
        """Create the necessary tables and triggers in the SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)

        # SQL to create the nedded tables
        create_table_sqls = {
            "embedding_model": """
        CREATE TABLE IF NOT EXISTS embedding_model (
            created_timestamp INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            PRIMARY KEY (embedding_model)
        )
        """,
            "messages": """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY NOT NULL,
            timestamp INTEGER NOT NULL,
            chat_model TEXT NOT NULL,
            message_exchange TEXT NOT NULL,
            embedding TEXT
        )
        """,
            "reply_audio_files": """
        CREATE TABLE IF NOT EXISTS reply_audio_files (
            id TEXT PRIMARY KEY NOT NULL,
            file_path TEXT NOT NULL,
            FOREIGN KEY (id) REFERENCES messages(id) ON DELETE CASCADE
        )
        """,
        }

        with conn:
            for table_name, table_create_sql in create_table_sqls.items():
                # Create tables
                conn.execute(table_create_sql)

                # Create triggers to prevent modification after insertion
                conn.execute(
                    f"""
                CREATE TRIGGER IF NOT EXISTS prevent_{table_name}_modification
                BEFORE UPDATE ON {table_name}
                BEGIN
                    SELECT RAISE(FAIL,  'Table "{table_name}": modification not allowed');
                END;
                """
                )

        # Close the connection to the database
        conn.close()

    def get_embedding_model(self):
        """Retrieve the database's embedding model.

        Returns:
            str: The embedding model or None if teh database is not yet initialised.
        """
        conn = sqlite3.connect(self.db_path)
        query = "SELECT embedding_model FROM embedding_model;"
        # Execute the query and fetch the result
        embedding_model = None
        with conn:
            cur = conn.cursor()
            cur.execute(query)
            result = cur.fetchone()
            embedding_model = result[0] if result else None

        conn.close()

        return embedding_model

    def insert_message_exchange(
        self, exchange_id, chat_model, message_exchange, embedding
    ):
        """Insert a message exchange into the database's 'messages' table.

        Args:
            exchange_id (str): The id of the message exchange.
            chat_model (str): The chat model.
            message_exchange: The message exchange.
            embedding: The embedding associated with the message exchange.

        Raises:
            ValueError: If the database already contains a different embedding model.
        """
        stored_embedding_model = self.get_embedding_model()
        if stored_embedding_model is None:
            self._init_database()
        elif stored_embedding_model != self.embedding_model:
            raise ValueError(
                "Database already contains a different embedding model: "
                f"{self.get_embedding_model()}.\n"
                "Cannot continue."
            )

        timestamp = int(datetime.datetime.utcnow().timestamp())
        message_exchange = json.dumps(message_exchange)
        embedding = json.dumps(embedding)
        conn = sqlite3.connect(self.db_path)
        sql = """
          INSERT INTO messages (id, timestamp, chat_model, message_exchange, embedding)
          VALUES (?, ?, ?, ?, ?)"""
        with conn:
            conn.execute(
                sql, (exchange_id, timestamp, chat_model, message_exchange, embedding)
            )
        conn.close()

    def insert_assistant_audio_file_path(
        self, exchange_id: str, file_path: Union[str, Path]
    ):
        """Insert the path to the assistant's reply audio file into the database.

        Args:
            exchange_id: The id of the message exchange.
            file_path: Path to the assistant's reply audio file.
        """
        file_path = file_path.as_posix()
        conn = sqlite3.connect(self.db_path)
        with conn:
            # Check if the corresponding id exists in the messages table
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM messages WHERE id=?", (exchange_id,))
            exists = cursor.fetchone() is not None
            if exists:
                # Insert into reply_audio_files
                cursor.execute(
                    "INSERT INTO reply_audio_files (id, file_path) VALUES (?, ?)",
                    (exchange_id, file_path),
                )
            else:
                logger.error("The corresponding id does not exist in the messages table")
        conn.close()

    def retrieve_history(self, exchange_id=None):
        """Retrieve data from all tables in the db combined in a single dataframe."""
        query = """
            SELECT messages.id,
                messages.timestamp,
                messages.chat_model,
                messages.message_exchange,
                reply_audio_files.file_path AS reply_audio_file_path,
                embedding
            FROM messages
            LEFT JOIN reply_audio_files
            ON messages.id = reply_audio_files.id
        """
        if exchange_id:
            query += f" WHERE messages.id = '{exchange_id}'"

        conn = sqlite3.connect(self.db_path)
        with conn:
            data_df = pd.read_sql_query(query, conn)
        conn.close()

        return data_df

    @property
    def n_entries(self):
        """Return the number of entries in the `messages` table."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT COUNT(*) FROM messages;"
        with conn:
            cur = conn.cursor()
            cur.execute(query)
            result = cur.fetchone()
        conn.close()
        return result[0]

    def _init_database(self):
        """Initialise the 'embedding_model' table in the database."""
        conn = sqlite3.connect(self.db_path)
        create_time = int(datetime.datetime.utcnow().timestamp())
        sql = "INSERT INTO embedding_model "
        sql += "(created_timestamp, embedding_model) VALUES (?, ?);"
        with conn:
            conn.execute(sql, (create_time, self.embedding_model))
        conn.close()
