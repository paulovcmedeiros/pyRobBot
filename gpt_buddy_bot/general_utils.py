from functools import wraps

import openai


class CannotConnectToApiError(Exception):
    """Error raised when the package cannot connect to the OpenAI API."""


def retry_api_call(max_n_attempts=5, auth_error_msg="Problems connecting to OpenAI API."):
    """Retry connecting to the API up to a maximum number of times."""

    def retry_api_call_decorator(function):
        """Wrap `function` and log beginning, exit and elapsed time."""

        @wraps(function)
        def wrapper(*args, **kwargs):
            n_attempts = 0
            success = False
            while not success:
                n_attempts += 1
                try:
                    function_rtn = function(*args, **kwargs)
                except (
                    openai.error.ServiceUnavailableError,
                    openai.error.Timeout,
                    openai.error.APIError,
                ) as error:
                    if n_attempts < max_n_attempts:
                        print(
                            f"\n    > {error}. "
                            + f"Making new attempt ({n_attempts+1}/{max_n_attempts})..."
                        )
                    else:
                        raise CannotConnectToApiError(auth_error_msg) from error
                else:
                    success = True

            return function_rtn

        return wrapper

    return retry_api_call_decorator
