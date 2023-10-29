#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

import openai


class GeneralConstants:
    _PACKAGE_TMPDIR = tempfile.TemporaryDirectory()
    PACKAGE_TMPDIR = Path(_PACKAGE_TMPDIR.name)
    EMBEDDINGS_FILE = PACKAGE_TMPDIR / "embeddings.csv"


# Initialize the OpenAI API client
openai.api_key = os.environ["OPENAI_API_KEY"]
