#!/usr/bin/env python3
"""Unnoficial OpenAI API UI and CLI tool."""
import os
import tempfile
import uuid
from importlib.metadata import version
from pathlib import Path

import openai


class GeneralConstants:
    """General constants for the package."""

    # Main package info
    RUN_ID = uuid.uuid4().hex
    PACKAGE_NAME = __name__
    VERSION = version(__name__)

    # Main package directories
    PACKAGE_DIRECTORY = Path(__file__).parent
    PACKAGE_CACHE_DIRECTORY = Path.home() / ".cache" / PACKAGE_NAME
    _PACKAGE_TMPDIR = tempfile.TemporaryDirectory()
    PACKAGE_TMPDIR = Path(_PACKAGE_TMPDIR.name)
    CHAT_CACHE_DIR = PACKAGE_CACHE_DIRECTORY / "chats"

    # Constants related to the app
    APP_NAME = "pyRobBot"
    APP_DIR = PACKAGE_DIRECTORY / "app"
    APP_PATH = APP_DIR / "app.py"
    PARSED_ARGS_FILE = PACKAGE_TMPDIR / f"parsed_args_{RUN_ID}.pkl"

    # Constants related to using the OpenAI API
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    TOKEN_USAGE_DATABASE = PACKAGE_CACHE_DIRECTORY / "token_usage.db"

    # Initialise the package's directories
    PACKAGE_TMPDIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CHAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Initialize the OpenAI API client
openai.api_key = GeneralConstants.OPENAI_API_KEY
