#!/usr/bin/env python3
"""Implementation of the Chat class."""
import contextlib
import json
import shutil
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import openai
from attr import dataclass
from loguru import logger
from pydub import AudioSegment
from tzlocal import get_localzone

from . import GeneralDefinitions
from .chat_configs import ChatOptions
from .chat_context import EmbeddingBasedChatContext, FullHistoryChatContext
from .general_utils import (
    AlternativeConstructors,
    ReachedMaxNumberOfAttemptsError,
    get_call_traceback,
)
from .internet_utils import websearch
from .openai_utils import OpenAiClientWrapper, make_api_chat_completion_call
from .sst_and_tts import SpeechToText, TextToSpeech
from .tokens import PRICE_PER_K_TOKENS_EMBEDDINGS, TokenUsageDatabase


@dataclass
class AssistantResponseChunk:
    """A chunk of the assistant's response."""

    exchange_id: str
    content: str
    chunk_type: str = "text"


class Chat(AlternativeConstructors):
    """Manages conversations with an AI chat model.

    This class encapsulates the chat behavior, including handling the chat context,
    managing cache directories, and interfacing with the OpenAI API for generating chat
    responses.
    """

    _translation_cache = defaultdict(dict)
    default_configs = ChatOptions()

    def __init__(
        self,
        openai_client: OpenAiClientWrapper = None,
        configs: ChatOptions = default_configs,
    ):
        """Initializes a chat instance.

        Args:
            configs (ChatOptions, optional): The configurations for this chat session.
            openai_client (openai.OpenAI, optional): An OpenAiClientWrapper instance.

        Raises:
            NotImplementedError: If the context model specified in configs is unknown.
        """
        self.id = str(uuid.uuid4())
        logger.trace(
            "Init chat {}, as requested by from <{}>", self.id, get_call_traceback()
        )
        logger.debug("Init chat {}", self.id)

        self._code_marker = "\uE001"  # TEST

        self._passed_configs = configs
        for field in self._passed_configs.model_fields:
            setattr(self, field, self._passed_configs[field])

        try:
            self.openai_client = (
                openai_client
                if openai_client is not None
                else OpenAiClientWrapper(
                    timeout=self.timeout, private_mode=self.private_mode
                )
            )
        except openai.OpenAIError as error:
            logger.opt(exception=True).debug(error)
            logger.error(
                "Cannot connect to OpenAI API. Please verify your API key. {}.", error
            )
            self.openai_client = None

    @property
    def base_directive(self):
        """Return the base directive for the LLM."""
        code_marker = self._code_marker
        local_datetime = datetime.now(get_localzone()).isoformat(timespec="seconds")
        msg_content = (
            f"Your name is {self.assistant_name}. Your model is {self.model}\n"
            f"You are a helpful assistant to {self.username}\n"
            f"You have internet access\n"
            f"You MUST ALWAYS write {code_marker} before AND after code blocks. Example: "
            f"```foo ... ``` MUST become {code_marker}```foo ... ```{code_marker}\n"
            f"The current city is {GeneralDefinitions.IPINFO['city']} in "
            f"{GeneralDefinitions.IPINFO['country_name']}\n"
            f"The local datetime is {local_datetime}\n"
            f"\nYou must follow all directives by {self.system_name} "
            f"unless otherwise instructed by {self.username}\n"
            f"You must reply in {self.language} unless otherwise instructed\n"
            "> If asked to look up online, web, internet etc, *you MUST ALWAYS AGREE*\n"
            "\n*IMPORTANT*: If you can't find information neither in your training data "
            "nor in the conversation context then, unless you have either already "
            f"looked up online or not understood {self.username}'s prompt, you must "
            "do as follows:\n"
            "  > Do *NOT* apologise nor say you are sorry nor give any excuses.\n"
            "  > Do *NOT* ask for permission to lookup online.\n"
            "  > STATE CLEARLY that you will look it up online.\n"
            "\n".join([f"{instruct.strip(' .')}." for instruct in self.ai_instructions])
        )
        return {"role": "system", "name": self.system_name, "content": msg_content}

    @property
    def configs(self):
        """Return the chat's configs after initialisation."""
        configs_dict = {}
        for field_name in self._passed_configs.model_fields:
            configs_dict[field_name] = getattr(self, field_name)
        return self._passed_configs.model_validate(configs_dict)

    @property
    def tmp_dir(self):
        """Return the temporary directory for the chat."""
        return Path(self._tmp_dir.name)

    @property
    def cache_dir(self):
        """Return the cache directory for this chat."""
        parent_dir = self.openai_client.get_cache_dir(private_mode=self.private_mode)
        directory = parent_dir / f"chat_{self.id}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def configs_file(self):
        """File to store the chat's configs."""
        return self.cache_dir / "configs.json"

    @property
    def context_file_path(self):
        """Return the path to the file that stores the chat context and history."""
        return self.cache_dir / "embeddings.db"

    @property
    def context_handler(self):
        """Return the chat's context handler."""
        if self.context_model == "full-history":
            return FullHistoryChatContext(parent_chat=self)

        if self.context_model in PRICE_PER_K_TOKENS_EMBEDDINGS:
            return EmbeddingBasedChatContext(parent_chat=self)

        raise NotImplementedError(f"Unknown context model: {self.context_model}")

    @property
    def token_usage_db(self):
        """Return the chat's token usage database."""
        return TokenUsageDatabase(fpath=self.cache_dir / "chat_token_usage.db")

    @property
    def general_token_usage_db(self):
        """Return the general token usage database for all chats.

        Even private-mode chats will use this database to keep track of total token usage.
        """
        general_cache_dir = self.openai_client.get_cache_dir(private_mode=False)
        return TokenUsageDatabase(fpath=general_cache_dir.parent / "token_usage.db")

    @property
    def metadata_file(self):
        """File to store the chat metadata."""
        return self.cache_dir / "metadata.json"

    @property
    def metadata(self):
        """Keep metadata associated with the chat."""
        try:
            _ = self._metadata
        except AttributeError:
            try:
                with open(self.metadata_file, "r") as f:
                    self._metadata = json.load(f)
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                self._metadata = {}
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = dict(value)

    def save_cache(self):
        """Store the chat's configs and metadata to the cache directory."""
        self.configs.export(self.configs_file)

        metadata = self.metadata  # Trigger loading metadata if not yet done
        metadata["chat_id"] = self.id
        with open(self.metadata_file, "w") as metadata_f:
            json.dump(metadata, metadata_f, indent=2)

    def clear_cache(self):
        """Remove the cache directory."""
        logger.debug("Clearing cache for chat {}", self.id)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def load_history(self):
        """Load chat history from cache."""
        return self.context_handler.load_history()

    @property
    def initial_greeting(self):
        """Return the initial greeting for the chat."""
        default_greeting = f"Hi! I'm {self.assistant_name}. How can I assist you?"
        user_set_greeting = False
        with contextlib.suppress(AttributeError):
            user_set_greeting = self._initial_greeting != ""

        if not user_set_greeting:
            self._initial_greeting = default_greeting

        custom_greeting = user_set_greeting and self._initial_greeting != default_greeting
        if custom_greeting or self.language[:2] != "en":
            self._initial_greeting = self._translate(self._initial_greeting)

        return self._initial_greeting

    @initial_greeting.setter
    def initial_greeting(self, value: str):
        self._initial_greeting = str(value).strip()

    def respond_user_prompt(self, prompt: str, **kwargs):
        """Respond to a user prompt."""
        yield from self._respond_prompt(prompt=prompt, role="user", **kwargs)

    def respond_system_prompt(
        self, prompt: str, add_to_history=False, skip_check=True, **kwargs
    ):
        """Respond to a system prompt."""
        for response_chunk in self._respond_prompt(
            prompt=prompt,
            role="system",
            add_to_history=add_to_history,
            skip_check=skip_check,
            **kwargs,
        ):
            yield response_chunk.content

    def yield_response_from_msg(
        self, prompt_msg: dict, add_to_history: bool = True, **kwargs
    ):
        """Yield response from a prompt message."""
        exchange_id = str(uuid.uuid4())
        code_marker = self._code_marker
        try:
            inside_code_block = False
            for answer_chunk in self._yield_response_from_msg(
                exchange_id=exchange_id,
                prompt_msg=prompt_msg,
                add_to_history=add_to_history,
                **kwargs,
            ):
                code_marker_detected = code_marker in answer_chunk
                inside_code_block = (code_marker_detected and not inside_code_block) or (
                    inside_code_block and not code_marker_detected
                )
                yield AssistantResponseChunk(
                    exchange_id=exchange_id,
                    content=answer_chunk.strip(code_marker),
                    chunk_type="code" if inside_code_block else "text",
                )

        except (ReachedMaxNumberOfAttemptsError, openai.OpenAIError) as error:
            yield self.response_failure_message(exchange_id=exchange_id, error=error)

    def start(self):
        """Start the chat."""
        # ruff: noqa: T201
        print(f"{self.assistant_name}> {self.initial_greeting}\n")
        try:
            while True:
                question = input(f"{self.username}> ").strip()
                if not question:
                    continue
                print(f"{self.assistant_name}> ", end="", flush=True)
                for chunk in self.respond_user_prompt(prompt=question):
                    print(chunk.content, end="", flush=True)
                print()
                print()
        except (KeyboardInterrupt, EOFError):
            print("", end="\r")
            logger.info("Leaving chat")

    def report_token_usage(self, report_current_chat=True, report_general: bool = False):
        """Report token usage and associated costs."""
        dfs = {}
        if report_general:
            dfs["All Recorded Chats"] = (
                self.general_token_usage_db.get_usage_balance_dataframe()
            )
        if report_current_chat:
            dfs["Current Chat"] = self.token_usage_db.get_usage_balance_dataframe()

        if dfs:
            for category, df in dfs.items():
                header = f"{df.attrs['description']}: {category}"
                table_separator = "=" * (len(header) + 4)
                print(table_separator)
                print(f"  {header}  ")
                print(table_separator)
                print(df)
                print()
            print(df.attrs["disclaimer"])

    def response_failure_message(
        self, exchange_id: Optional[str] = "", error: Optional[Exception] = None
    ):
        """Return the error message errors getting a response."""
        msg = "Could not get a response right now."
        if error is not None:
            msg += f" The reason seems to be: {error} "
            msg += "Please check your connection or OpenAI API key."
            logger.opt(exception=True).debug(error)
        return AssistantResponseChunk(exchange_id=exchange_id, content=msg)

    def stt(self, speech: AudioSegment):
        """Convert audio to text."""
        return SpeechToText(
            speech=speech,
            openai_client=self.openai_client,
            engine=self.stt_engine,
            language=self.language,
            timeout=self.timeout,
            general_token_usage_db=self.general_token_usage_db,
            token_usage_db=self.token_usage_db,
        )

    def tts(self, text: str):
        """Convert text to audio."""
        return TextToSpeech(
            text=text,
            openai_client=self.openai_client,
            language=self.language,
            engine=self.tts_engine,
            openai_tts_voice=self.openai_tts_voice,
            timeout=self.timeout,
            general_token_usage_db=self.general_token_usage_db,
            token_usage_db=self.token_usage_db,
        )

    def _yield_response_from_msg(
        self,
        exchange_id,
        prompt_msg: dict,
        add_to_history: bool = True,
        skip_check: bool = False,
    ):
        """Yield response from a prompt message (lower level interface)."""
        # Get appropriate context for prompt from the context handler
        context = self.context_handler.get_context(msg=prompt_msg)

        # Make API request and yield response chunks
        full_reply_content = ""
        for chunk in make_api_chat_completion_call(
            conversation=[self.base_directive, *context, prompt_msg], chat_obj=self
        ):
            full_reply_content += chunk.strip(self._code_marker)
            yield chunk

        if not skip_check:
            last_msg_exchange = (
                f"`user` says: {prompt_msg['content']}\n"
                f"`you` replies: {full_reply_content}"
            )
            system_check_msg = (
                "Consider the following dialogue between `user` and `you` "
                "AND NOTHING MORE:\n\n"
                f"{last_msg_exchange}\n\n"
                "Now answer the following question using only 'yes' or 'no':\n"
                "Were `you` able to provide a good answer the `user`s prompt, without "
                "neither `you` nor `user` asking or implying the need or intention to "
                "perform a search or lookup online, on the web or the internet?\n"
            )

            reply = "".join(self.respond_system_prompt(prompt=system_check_msg))
            reply = reply.strip(".' ").lower()
            if ("no" in reply) or (self._translate("no") in reply):
                instructions_for_web_search = (
                    "You are a professional web searcher. You will be presented with a "
                    "dialogue between `user` and `you`. Considering the dialogue and "
                    "relevant previous messages, write "
                    "the best short web search query to look for an answer to the "
                    "`user`'s prompt. You MUST follow the rules below:\n"
                    "* Write *only the query* and nothing else\n"
                    "* DO NOT RESTRICT the search to any particular website "
                    "unless otherwise instructed\n"
                    "* You MUST reply in the `user`'s language unless otherwise asked\n\n"
                    "The `dialogue` is:"
                )
                instructions_for_web_search += f"\n\n{last_msg_exchange}"
                internet_query = "".join(
                    self.respond_system_prompt(prompt=instructions_for_web_search)
                )
                yield "\n\n" + self._translate(
                    "Searching the web now. My search is: "
                ) + f" '{internet_query}'..."
                web_results_json_dumps = "\n\n".join(
                    json.dumps(result, indent=2) for result in websearch(internet_query)
                )
                if web_results_json_dumps:
                    logger.opt(colors=True).debug(
                        "Web search rtn: <yellow>{}</yellow>...", web_results_json_dumps
                    )
                    original_prompt = prompt_msg["content"]
                    prompt = (
                        "You are a talented data analyst, "
                        "capable of summarising any information, even complex `json`. "
                        "You will be shown a `json` and a `prompt`. Your task is to "
                        "summarise the `json` to answer the `prompt`. "
                        "You MUST follow the rules below:\n\n"
                        "* *ALWAYS* provide a meaningful summary to the the `json`\n"
                        "* *Do NOT include links* or anything a human can't pronounce, "
                        "unless otherwise instructed\n"
                        "* Prefer searches without quotes but use them if needed\n"
                        "* Answer in human language (i.e., no json, etc)\n"
                        "* Answer in the `user`'s language unless otherwise asked\n"
                        "* Make sure to point out that the information is from a quick "
                        "web search and may be innacurate\n"
                        "* Mention the sources shortly WITHOUT MENTIONING WEB LINKS\n\n"
                        "The `json` and the `prompt` are presented below:\n"
                    )
                    prompt += f"\n```json\n{web_results_json_dumps}\n```\n"
                    prompt += f"\n`prompt`: '{original_prompt}'"

                    yield "\n\n" + self._translate(
                        " I've got some results. Let me summarise them for you..."
                    )

                    full_reply_content += " "
                    yield "\n\n"
                    for chunk in self.respond_system_prompt(prompt=prompt):
                        full_reply_content += chunk.strip(self._code_marker)
                        yield chunk
                else:
                    yield self._translate(
                        "Sorry, but I couldn't find anything on the web this time."
                    )

        if add_to_history:
            # Put current chat exchange in context handler's history
            self.context_handler.add_to_history(
                exchange_id=exchange_id,
                msg_list=[
                    prompt_msg,
                    {"role": "assistant", "content": full_reply_content},
                ],
            )

    def _respond_prompt(self, prompt: str, role: str, **kwargs):
        prompt_as_msg = {"role": role.lower().strip(), "content": prompt.strip()}
        yield from self.yield_response_from_msg(prompt_as_msg, **kwargs)

    def _translate(self, text):
        lang = self.language

        cached_translation = type(self)._translation_cache[text].get(lang)  # noqa SLF001
        if cached_translation:
            return cached_translation

        logger.debug("Processing translation of '{}' to '{}'...", text, lang)
        translation_prompt = (
            f"Translate the text between triple quotes below to {lang}. "
            "DO NOT WRITE ANYTHING ELSE. Only the translation. "
            f"If the text is already in {lang}, then don't translate. Just return ''.\n"
            f"'''{text}'''"
        )
        translation = "".join(self.respond_system_prompt(prompt=translation_prompt))

        translation = translation.strip(" '\"")
        if not translation.strip():
            translation = text.strip()

        logger.debug("Translated '{}' to '{}' as '{}'", text, lang, translation)
        type(self)._translation_cache[text][lang] = translation  # noqa: SLF001
        type(self)._translation_cache[translation][lang] = translation  # noqa: SLF001

        return translation

    def __del__(self):
        """Delete the chat instance."""
        logger.debug("Deleting chat {}", self.id)
        chat_started = self.context_handler.database.n_entries > 0
        if self.private_mode or not chat_started:
            self.clear_cache()
        else:
            self.save_cache()
            self.clear_cache()
