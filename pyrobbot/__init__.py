#!/usr/bin/env python3
"""Unnoficial OpenAI API UI and CLI tool."""
import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import openai


@dataclass
class GeneralDefinitions:
    """General definitions for the package."""

    # Keep track of the OpenAI API key available to the package at initialization
    SYSTEM_ENV_OPENAI_API_KEY: str = None

    # Main package info
    RUN_ID = uuid.uuid4().hex
    PACKAGE_NAME = __name__
    VERSION = version(__name__)

    # Main package directories
    PACKAGE_DIRECTORY = Path(__file__).parent
    _PACKAGE_TMPDIR = tempfile.TemporaryDirectory()
    PACKAGE_TMPDIR = Path(_PACKAGE_TMPDIR.name)

    # Constants related to the app
    APP_NAME = "pyRobBot"
    APP_DIR = PACKAGE_DIRECTORY / "app"
    APP_PATH = APP_DIR / "app.py"
    PARSED_ARGS_FILE = PACKAGE_TMPDIR / f"parsed_args_{RUN_ID}.pkl"

    @staticmethod
    def openai_key_hash():
        """Return a hash of the OpenAI API key."""
        if openai.api_key is None:
            return "demo"
        return hashlib.sha256(openai.api_key.encode("utf-8")).hexdigest()

    @property
    def package_cache_directory(self):
        """Return the path to the package's cache directory."""
        return (
            Path.home() / ".cache" / self.PACKAGE_NAME / f"user_{self.openai_key_hash()}"
        )

    @property
    def chat_cache_dir(self):
        """Return the path to the package's cache directory."""
        return self.package_cache_directory / "chats"

    @property
    def general_token_usage_db_path(self):
        """Return the path to the package's token usage database."""
        return self.package_cache_directory / "token_usage.db"


GeneralConstants = GeneralDefinitions(
    SYSTEM_ENV_OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
)

# Initialize the OpenAI API client
openai.api_key = GeneralConstants.SYSTEM_ENV_OPENAI_API_KEY
