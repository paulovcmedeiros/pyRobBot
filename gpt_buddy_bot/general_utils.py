import inspect
import time
from functools import wraps

import openai


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
                else:
                    success = True

        return wrapper_generator_f if inspect.isgeneratorfunction(function) else wrapper_f

    return retry_api_call_decorator
