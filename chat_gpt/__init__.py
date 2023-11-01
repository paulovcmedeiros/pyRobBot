#!/usr/bin/env python3
import os
import tempfile
import uuid
from importlib.metadata import version
from pathlib import Path

import openai


class GeneralConstants:
    PACKAGE_NAME = __name__
    VERSION = version(__name__)
    PACKAGE_DIRECTORY = Path(__file__).parent
    RUN_ID = uuid.uuid4().hex
    PACKAGE_CACHE_DIRECTORY = Path.home() / ".cache" / PACKAGE_NAME
    _PACKAGE_TMPDIR = tempfile.TemporaryDirectory()
    PACKAGE_TMPDIR = Path(_PACKAGE_TMPDIR.name)
    PARSED_ARGS_FILE = PACKAGE_TMPDIR / f"parsed_args_{RUN_ID}.pkl"
    TOKEN_USAGE_DATABASE = PACKAGE_CACHE_DIRECTORY / "token_usage.db"

    PACKAGE_TMPDIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)


# Initialize the OpenAI API client
openai.api_key = os.environ["OPENAI_API_KEY"]
