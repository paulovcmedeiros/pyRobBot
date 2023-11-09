import inspect
import time
from functools import wraps
from typing import TYPE_CHECKING

import openai

from .chat_configs import OpenAiApiCallOptions

if TYPE_CHECKING:
    from .chat import Chat


class CannotConnectToApiError(Exception):
    """Error raised when the package cannot connect to the OpenAI API."""


def retry_api_call(max_n_attempts=5, auth_error_msg="Problems connecting to OpenAI API."):
    """Retry connecting to the API up to a maximum number of times."""
    handled_exceptions = (
        openai.error.ServiceUnavailableError,
        openai.error.Timeout,
        openai.error.APIError,
    )

    def on_error(error, n_attempts):
        if n_attempts < max_n_attempts:
            print(
                f"\n    > {error}. "
                + f"Making new attempt ({n_attempts+1}/{max_n_attempts})..."
            )
            time.sleep(1)
        else:
            raise CannotConnectToApiError(auth_error_msg) from error

    def retry_api_call_decorator(function):
        """Wrap `function` and log beginning, exit and elapsed time."""

        @wraps(function)
        def wrapper_f(*args, **kwargs):
            n_attempts = 0
            while True:
                n_attempts += 1
                try:
                    return function(*args, **kwargs)
                except handled_exceptions as error:
                    on_error(error=error, n_attempts=n_attempts)
                except openai.error.AuthenticationError as error:
                    raise CannotConnectToApiError(auth_error_msg) from error

        @wraps(function)
        def wrapper_generator_f(*args, **kwargs):
            n_attempts = 0
            success = False
            while not success:
                n_attempts += 1
                try:
                    yield from function(*args, **kwargs)
                except handled_exceptions as error:
                    on_error(error=error, n_attempts=n_attempts)
                except openai.error.AuthenticationError as error:
                    raise CannotConnectToApiError(auth_error_msg) from error
                else:
                    success = True

        return wrapper_generator_f if inspect.isgeneratorfunction(function) else wrapper_f

    return retry_api_call_decorator


def make_api_chat_completion_call(conversation: list, chat_obj: "Chat"):
    api_call_args = {}
    for field in OpenAiApiCallOptions.model_fields:
        if getattr(chat_obj, field) is not None:
            api_call_args[field] = getattr(chat_obj, field)

    @retry_api_call(auth_error_msg=chat_obj.api_connection_error_msg)
    def stream_reply(conversation, **api_call_args):
        for completion_chunk in openai.ChatCompletion.create(
            messages=conversation, stream=True, **api_call_args
        ):
            reply_chunk = getattr(completion_chunk.choices[0].delta, "content", "")
            yield reply_chunk

    yield from stream_reply(conversation, **api_call_args)
