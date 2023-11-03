#!/usr/bin/env python3
"""Registration and validation of options."""
from functools import reduce
from getpass import getuser
from pathlib import Path
from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated

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
        annotation = cls._get_field_param(field=field, param="annotation")
        if isinstance(annotation, type):
            return annotation

    @classmethod
    def get_default(cls, field: str):
        """Return allowed value(s) for `field`."""
        return cls._get_field_param(field=field, param="default")

    @classmethod
    def get_description(cls, field: str):
        return cls._get_field_param(field=field, param="description")

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


openai_url = "https://platform.openai.com/docs/api-reference/chat/create#chat-create"


class ChatOptions(BaseConfigModel):
    """Model for the chat's configuration options."""

    model: Literal["gpt-3.5-turbo", "gpt-4"] = Field(
        default="gpt-3.5-turbo", description="OpenAI API engine to use for completion"
    )
    username: str = Field(default=getuser(), description="Name of the chat's user")
    assistant_name: str = Field(
        default="Based on model", description="Name of the chat's assistant"
    )
    system_name: str = Field(
        default=GeneralConstants.PACKAGE_NAME, description="Name of the chat's system"
    )
    context_model: Literal["text-embedding-ada-002", None] = Field(
        default="text-embedding-ada-002",
        description="OpenAI API engine to use for embedding",
    )
    ai_instructions: tuple[str, ...] = Field(
        default=("Answer with the fewest tokens possible.",),
        description="Initial instructions for the AI",
    )
    max_tokens: Optional[
        Annotated[int, Field(gt=0, description=f"See <{openai_url}-max_tokens>")]
    ] = None
    frequency_penalty: Optional[
        Annotated[
            float,
            Field(ge=-2.0, le=2.0, description=f"See <{openai_url}-frequency_penalty>"),
        ]
    ] = None
    presence_penalty: Optional[
        Annotated[
            float,
            Field(ge=-2.0, le=2.0, description=f"See <{openai_url}-presence_penalty>"),
        ]
    ] = None
    temperature: Optional[
        Annotated[
            float,
            Field(ge=0.0, le=2.0, description=f"See <{openai_url}-temperature>"),
        ]
    ] = None
    top_p: Optional[
        Annotated[
            float,
            Field(defaut=None, ge=0.0, le=1.0, description=f"See <{openai_url}-top_p>"),
        ]
    ] = None
    token_usage_db_path: Path = Field(
        default=GeneralConstants.TOKEN_USAGE_DATABASE,
        description="Path to the token usage database",
    )
    report_accounting_when_done: bool = Field(
        default=False, description="Report estimated costs when done with the chat."
    )

    @validator("assistant_name", always=True)
    def get_address(cls, assistant_name: str, values: dict[str, Any]) -> str:
        assistant_name = assistant_name.lower().strip()
        if assistant_name == "based on model":
            return f"chat_{values.get('model', 'assistant').replace('.', '_')}"
        return assistant_name


DEFAULT_CHAT_OPTIONS = ChatOptions()
