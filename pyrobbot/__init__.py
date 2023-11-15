#!/usr/bin/env python3
"""Unnoficial OpenAI API UI and CLI tool."""
import hashlib
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from importlib.metadata import metadata, version
from pathlib import Path

import openai
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("LOGLEVEL", os.environ.get("LOGURU_LEVEL", "INFO")),
)


@dataclass
class GeneralDefinitions:
    """General definitions for the package."""

    # Keep track of the OpenAI API key available to the package at initialization
    SYSTEM_ENV_OPENAI_API_KEY: str = None

    # Main package info
    RUN_ID = uuid.uuid4().hex
    PACKAGE_NAME = __name__
    VERSION = version(__name__)
    PACKAGE_DESCRIPTION = metadata(__name__)["Summary"]

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
        """Return the path to the package's general cache directory."""
        return Path.home() / ".cache" / self.PACKAGE_NAME

    @property
    def current_user_cache_dir(self):
        """Return the directory where cache info for the current user is stored."""
        return self.package_cache_directory / f"user_{self.openai_key_hash()}"

    @property
    def chats_storage_dir(self):
        """Return the directory where the current user's chats are stored."""
        return self.current_user_cache_dir / "chats"


GeneralConstants = GeneralDefinitions(
    SYSTEM_ENV_OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
)

# Initialize the OpenAI API client

