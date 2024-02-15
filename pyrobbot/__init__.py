#!/usr/bin/env python3
"""Unnoficial OpenAI API UI and CLI tool."""
import hashlib
import os
import sys
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import metadata, version
from pathlib import Path

import ipinfo
import requests
from loguru import logger
from openai import OpenAI

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("LOGLEVEL", os.environ.get("LOGURU_LEVEL", "INFO")),
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


@dataclass
class GeneralDefinitions:
    """General definitions for the package."""

    # Main package info
    RUN_ID = uuid.uuid4().hex
    PACKAGE_NAME = __name__
    VERSION = version(__name__)
    PACKAGE_DESCRIPTION = metadata(__name__)["Summary"]

    # Main package directories
    PACKAGE_DIRECTORY = Path(__file__).parent
    PACKAGE_CACHE_DIRECTORY = Path.home() / ".cache" / PACKAGE_NAME
    _PACKAGE_TMPDIR = tempfile.TemporaryDirectory()
    PACKAGE_TMPDIR = Path(_PACKAGE_TMPDIR.name)

    # Constants related to the app
    APP_NAME = "pyRobBot"
    APP_DIR = PACKAGE_DIRECTORY / "app"
    APP_PATH = APP_DIR / "app.py"
    PARSED_ARGS_FILE = PACKAGE_TMPDIR / f"parsed_args_{RUN_ID}.pkl"

    # Location info
    try:
        IPINFO = ipinfo.getHandler().getDetails().all
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
        IPINFO = defaultdict(lambda: "unknown")

    @classmethod
    def get_openai_client_cache_dir(cls, openai_client: OpenAI = None):
        """Return the directory where chats using openai_client will be stored."""
        if openai_client is None:
            return cls.PACKAGE_CACHE_DIRECTORY / "user_demo"
        key_hash = hashlib.sha256(openai_client.api_key.encode("utf-8")).hexdigest()
        return cls.PACKAGE_CACHE_DIRECTORY / f"user_{key_hash}"
