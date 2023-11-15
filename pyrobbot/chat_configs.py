#!/usr/bin/env python3
"""Registration and validation of options."""
import argparse
import json
import types
import typing
from getpass import getuser
from pathlib import Path
from typing import Literal, Optional, get_args, get_origin

from pydantic import BaseModel, Field

from pyrobbot import GeneralConstants


class BaseConfigModel(BaseModel, extra="forbid"):
    """Base model for configuring options."""

    @classmethod
    def get_allowed_values(cls, field: str):
        """Return a tuple of allowed values for `field`."""
        annotation = cls._get_field_param(field=field, param="annotation")
        if isinstance(annotation, type(Literal[""])):
            return get_args(annotation)
        return None

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
        return None

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
        """Make possible to retrieve values as in a dict."""
        try:
            return getattr(self, item)
        except AttributeError as error:
            raise KeyError(item) from error

    def export(self, fpath: Path):
        """Export the model's data to a file."""
        with open(fpath, "w") as configs_file:
            configs_file.write(self.model_dump_json(indent=2, exclude_unset=True))

    @classmethod
    def from_file(cls, fpath: Path):
        """Return an instance of the class given configs stored in a json file."""
        with open(fpath, "r") as configs_file:
            return cls.model_validate(json.load(configs_file))


class OpenAiApiCallOptions(BaseConfigModel):
    """Model for configuring options for OpenAI API calls."""

    _openai_url = "https://platform.openai.com/docs/api-reference/chat/create#chat-create"
    _models_url = "https://platform.openai.com/docs/models"

    model: Literal[
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",  # Will point to gpt-3.5-turbo-1106 starting Dec 11, 2023
        "gpt-3.5-turbo",  # Will point to gpt-3.5-turbo-1106 starting Dec 11, 2023
        "gpt-4-1106-preview",
        "gpt-4",
    ] = Field(
        default="gpt-3.5-turbo-1106",
        description=f"OpenAI LLM model to use. See {_openai_url}-model and {_models_url}",
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
    timeout: Optional[float] = Field(
        default=10.0, gt=0.0, description="Timeout for API requests in seconds"
    )


class ChatOptions(OpenAiApiCallOptions):
    """Model for the chat's configuration options."""

    username: str = Field(default=getuser(), description="Name of the chat's user")
    assistant_name: str = Field(default="Rob", description="Name of the chat's assistant")
    system_name: str = Field(
        default=f"{GeneralConstants.PACKAGE_NAME}_system",
        description="Name of the chat's system",
    )
    context_model: Literal["text-embedding-ada-002", "full-history"] = Field(
        default="text-embedding-ada-002",
        description=(
            "Model to use for chat context (~memory). "
            + "Once picked, it cannot be changed."
        ),
        json_schema_extra={"frozen": True},
    )
    ai_instructions: tuple[str, ...] = Field(
        default=(
            "You answer correctly.",
            "You do not lie.",
        ),
        description="Initial instructions for the AI",
    )
    initial_greeting: Optional[str] = Field(
        default="", description="Initial greeting given by the assistant"
    )
    api_connection_max_n_attempts: int = Field(
        default=5,
        gt=0,
        description="Maximum number of attempts to connect to the OpenAI API",
    )
    language_speech: str = Field(
        default="en", description="Language for text to speech/speech to text"
    )
    private_mode: Optional[bool] = Field(
        default=None,
        description="Toggle private mode. If set to `True`, the chat will not "
        + "be logged and the chat history will not be saved.",
    )
