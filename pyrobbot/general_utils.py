"""General utility functions and classes."""

import difflib
import inspect
import json
import re
import time
from functools import wraps
from pathlib import Path
from typing import Optional

import httpx
import openai
from loguru import logger
from pydub import AudioSegment
from pydub.silence import detect_leading_silence


class ReachedMaxNumberOfAttemptsError(Exception):
    """Error raised when the max number of attempts has been reached."""


def _get_lower_alphanumeric(string: str):
    """Return a string with only lowercase alphanumeric characters."""
    return re.sub("[^0-9a-zA-Z]+", " ", string.strip().lower())


def str2_minus_str1(str1: str, str2: str):
    """Return the words in str2 that are not in str1."""
    output_list = [diff for diff in difflib.ndiff(str1, str2) if diff[0] == "+"]
    str_diff = "".join(el.replace("+ ", "") for el in output_list if el.startswith("+"))
    return str_diff


def get_call_traceback(depth=5):
    """Get the traceback of the call to the function."""
    curframe = inspect.currentframe()
    callframe = inspect.getouterframes(curframe)
    call_path = []
    for iframe, frame in enumerate(callframe):
        fpath = frame.filename
        lineno = frame.lineno
        function = frame.function
        code_context = frame.code_context[0].strip()
        call_path.append(
            {
                "fpath": fpath,
                "lineno": lineno,
                "function": function,
                "code_context": code_context,
            }
        )
        if iframe == depth:
            break
    return call_path


def trim_beginning(audio: AudioSegment, **kwargs):
    """Trim the beginning of the audio to remove silence."""
    beginning = detect_leading_silence(audio, **kwargs)
    return audio[beginning:]


def trim_ending(audio: AudioSegment, **kwargs):
    """Trim the ending of the audio to remove silence."""
    audio = trim_beginning(audio.reverse(), **kwargs)
    return audio.reverse()


def trim_silence(audio: AudioSegment, **kwargs):
    """Trim the silence from the beginning and ending of the audio."""
    kwargs["silence_threshold"] = kwargs.get("silence_threshold", -40.0)
    audio = trim_beginning(audio, **kwargs)
    return trim_ending(audio, **kwargs)


def retry(
    max_n_attempts: int = 5,
    handled_errors: tuple[Exception, ...] = (
        openai.APITimeoutError,
        httpx.HTTPError,
        RuntimeError,
    ),
    error_msg: Optional[str] = None,
):
    """Retry executing the decorated function/generator."""

    def retry_or_fail(error):
        """Decide whether to retry or fail based on the number of attempts."""
        retry_or_fail.execution_count = getattr(retry_or_fail, "execution_count", 0) + 1

        if retry_or_fail.execution_count < max_n_attempts:
            logger.warning(
                "{}. Making new attempt ({}/{})...",
                error,
                retry_or_fail.execution_count + 1,
                max_n_attempts,
            )
            time.sleep(1)
        else:
            raise ReachedMaxNumberOfAttemptsError(error_msg) from error

    def retry_decorator(function):
        """Wrap `function`."""

        @wraps(function)
        def wrapper_f(*args, **kwargs):
            while True:
                try:
                    return function(*args, **kwargs)
                except handled_errors as error:  # noqa: PERF203
                    retry_or_fail(error=error)

        @wraps(function)
        def wrapper_generator_f(*args, **kwargs):
            success = False
            while not success:
                try:
                    yield from function(*args, **kwargs)
                except handled_errors as error:  # noqa: PERF203
                    retry_or_fail(error=error)
                else:
                    success = True

        return wrapper_generator_f if inspect.isgeneratorfunction(function) else wrapper_f

    return retry_decorator


class AlternativeConstructors:
    """Mixin class for alternative constructors."""

    @classmethod
    def from_dict(cls, configs: dict, **kwargs):
        """Creates an instance from a configuration dictionary.

        Converts the configuration dictionary into a instance of this class
        and uses it to instantiate the Chat class.

        Args:
            configs (dict): The configuration options as a dictionary.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            cls: An instance of Chat initialized with the given configurations.
        """
        return cls(configs=cls.default_configs.model_validate(configs), **kwargs)

    @classmethod
    def from_cli_args(cls, cli_args, **kwargs):
        """Creates an instance from CLI arguments.

        Extracts relevant options from the CLI arguments and initializes a class instance
        with them.

        Args:
            cli_args: The command line arguments.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            cls: An instance of the class initialized with CLI-specified configurations.
        """
        chat_opts = {
            k: v
            for k, v in vars(cli_args).items()
            if k in cls.default_configs.model_fields and v is not None
        }
        return cls.from_dict(chat_opts, **kwargs)

    @classmethod
    def from_cache(cls, cache_dir: Path, **kwargs):
        """Loads an instance from a cache directory.

        Args:
            cache_dir (Path): The path to the cache directory.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            cls: An instance of the class loaded with cached configurations and metadata.
        """
        try:
            with open(cache_dir / "configs.json", "r") as configs_f:
                new_configs = json.load(configs_f)
        except FileNotFoundError:
            logger.warning(
                "Could not find config file in cache directory <{}>. "
                + "Creating {} with default configs.",
                cache_dir,
                cls.__name__,
            )
            new_configs = cls.default_configs.model_dump()

        try:
            with open(cache_dir / "metadata.json", "r") as metadata_f:
                new_metadata = json.load(metadata_f)
        except FileNotFoundError:
            logger.warning(
                "Could not find metadata file in cache directory <{}>. "
                + "Creating {} with default metadata.",
                cache_dir,
                cls.__name__,
            )
            new_metadata = None

        new = cls.from_dict(new_configs, **kwargs)
        if new_metadata is not None:
            new.metadata = new_metadata
            logger.debug(
                "Reseting chat_id from cache: {} --> {}.",
                new.id,
                new.metadata["chat_id"],
            )
            new.id = new.metadata["chat_id"]

        return new
