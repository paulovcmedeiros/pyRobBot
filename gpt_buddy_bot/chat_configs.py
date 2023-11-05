#!/usr/bin/env python3
"""Registration and validation of options."""
import argparse
import os
import types
import typing
from functools import reduce
from getpass import getuser
from pathlib import Path
from typing import Literal, Optional, get_args, get_origin

import openai
from pydantic import BaseModel, Field, SecretStr

from gpt_buddy_bot import GeneralConstants


class BaseConfigModel(BaseModel):
    @classmethod
    def get_allowed_values(cls, field: str):
        """Return a tuple of allowed values for `field`."""
        annotation = cls._get_field_param(field=field, param="annotation")
        if isinstance(annotation, type(Literal[""])):
            return get_args(annotation)

    @classmethod
    def get_type(cls, field: str):
        """Return type of `field`."""
        type_hint = typing.get_type_hints(cls)[field]
        if isinstance(type_hint, type):
            if isinstance(type_hint, types.GenericAlias):
                return get_origin(type_hint)
            return type_hint
        type_hint_first_arg = get_args(type_hint)[0]
        if isinstance(type_hint_first_arg, type):
            return type_hint_first_arg

    @classmethod
    def get_default(cls, field: str):
        """Return allowed value(s) for `field`."""
        return cls.model_fields[field].get_default()

    @classmethod
    def get_description(cls, field: str):
        """Return description of `field`."""
        return cls._get_field_param(field=field, param="description")

    @classmethod
    def from_cli_args(cls, cli_args: argparse.Namespace):
        """Return an instance of the class from CLI args."""
        relevant_args = {
            k: v
            for k, v in vars(cli_args).items()
            if k in cls.model_fields and v is not None
        }
        return cls.model_validate(relevant_args)

    @classmethod
    def _get_field_param(cls, field: str, param: str):
        """Return param `param` of field `field`."""
        return getattr(cls.model_fields[field], param, None)

    def __getitem__(self, item):
        """Get items from container.

        The behaviour is similar to a `dict`, except for the fact that
        `self["A.B.C.D. ..."]` will behave like `self["A"]["B"]["C"]["D"][...]`.

        Args:
            item (str): Item to be retrieved. Use dot-separated keys to retrieve a nested
                item in one go.

        Raises:
            KeyError: If the item is not found.

        Returns:
            Any: Value of the item.
        """
        try:
            # Try regular getitem first in case "A.B. ... C" is actually a single key
            return getattr(self, item)
        except AttributeError:
            try:
                return reduce(getattr, item.split("."), self)
            except AttributeError as error:
                raise KeyError(item) from error


class OpenAiApiCallOptions(BaseConfigModel):
    _openai_url = "https://platform.openai.com/docs/api-reference/chat/create#chat-create"

    model: Literal["gpt-3.5-turbo", "gpt-4"] = Field(
        default="gpt-3.5-turbo",
        description=f"OpenAI LLM model to use. See {_openai_url}-model",
    )
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description=f"See <{_openai_url}-max_tokens>"
    )
    presence_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description=f"See <{_openai_url}-presence_penalty>"
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=f"See <{_openai_url}-frequency_penalty>",
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description=f"See <{_openai_url}-temperature>"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description=f"See <{_openai_url}-top_p>"
    )
    request_timeout: Optional[float] = Field(
        default=10.0, gt=0.0, description="Timeout for API requests in seconds"
    )


class ChatOptions(OpenAiApiCallOptions):
    """Model for the chat's configuration options."""

    username: str = Field(default=getuser(), description="Name of the chat's user")
    assistant_name: str = Field(
        default=GeneralConstants.APP_NAME, description="Name of the chat's assistant"
    )
    system_name: str = Field(
        default=f"{GeneralConstants.PACKAGE_NAME}_system",
        description="Name of the chat's system",
    )
    context_model: Literal["text-embedding-ada-002", None] = Field(
        default="text-embedding-ada-002",
        description="OpenAI API model to use for embedding",
    )
    context_file_path: Optional[Path] = Field(
        default=None,
        description="Path to the file to read/write the chat context from/to.",
    )
    ai_instructions: tuple[str, ...] = Field(
        default=(
            "You answer correctly.",
            "You do not lie.",
            "You answer with the fewest tokens possible.",
        ),
        description="Initial instructions for the AI",
    )
    token_usage_db_path: Optional[Path] = Field(
        default=GeneralConstants.TOKEN_USAGE_DATABASE,
        description="Path to the token usage database",
    )
    report_accounting_when_done: Optional[bool] = Field(
        default=False, description="Report estimated costs when done with the chat."
    )
