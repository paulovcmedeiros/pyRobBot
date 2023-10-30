#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

import openai


class GeneralConstants:
    _PACKAGE_TMPDIR = tempfile.TemporaryDirectory()
    PACKAGE_TMPDIR = Path(_PACKAGE_TMPDIR.name)
    EMBEDDINGS_FILE = PACKAGE_TMPDIR / "embeddings.csv"
    TOKEN_USAGE_DATABASE = Path.home() / ".cache" / "chat_gpt" / "token_usage.db"


# Initialize the OpenAI API client
openai.api_key = os.environ["OPENAI_API_KEY"]
