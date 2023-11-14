#!/usr/bin/env python3
"""Implementation of the Chat class."""
import json
import shutil
import uuid
from pathlib import Path

from loguru import logger

from . import GeneralConstants
from .chat_configs import ChatOptions
from .chat_context import EmbeddingBasedChatContext, FullHistoryChatContext
from .openai_utils import CannotConnectToApiError, make_api_chat_completion_call
from .tokens import TokenUsageDatabase


class Chat:
    """Manages conversations with an AI chat model.

    This class encapsulates the chat behavior, including handling the chat context,
    managing cache directories, and interfacing with the OpenAI API for generating chat
    responses.
    """

    def __init__(self, configs: ChatOptions = None):
        """Initializes a chat instance.

        Args:
            configs (ChatOptions, optional): The configurations for this chat session.

        Raises:
            NotImplementedError: If the context model specified in configs is unknown.
        """
        self.id = str(uuid.uuid4())
        self.initial_openai_key_hash = GeneralConstants.openai_key_hash()

        if configs is None:
            configs = ChatOptions()

        self._passed_configs = configs
        for field in self._passed_configs.model_fields:
            setattr(self, field, self._passed_configs[field])

    @property
    def base_directive(self):
        """Return the base directive for the LLM."""
        msg_content = " ".join(
            [
                instruction.strip()
                for instruction in [
                    f"You are {self.assistant_name} (model {self.model}).",
                    f"You are a helpful assistant to {self.username}.",
                    " ".join(
                        [f"{instruct.strip(' .')}." for instruct in self.ai_instructions]
                    ),
                    f"You must remember and follow all directives by {self.system_name} ",
                    f"unless otherwise instructed by {self.username}.",
                ]
                if instruction.strip()
            ]
        )
        return {"role": "system", "name": self.system_name, "content": msg_content}

    @property
    def configs(self):
        """Return the chat's configs after initialisation."""
        configs_dict = {}
        for field_name in ChatOptions.model_fields:
            configs_dict[field_name] = getattr(self, field_name)
        return ChatOptions.model_validate(configs_dict)

    @property
    def user_cache_dir(self):
        """Return the general-purpose cache directory assigned to the user."""
        return GeneralConstants.current_user_cache_dir

    @property
    def cache_dir(self):
        """Return the cache directory for this chat."""
        directory = self.user_cache_dir / f"chat_{self.id}"
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

        if self.context_model == "text-embedding-ada-002":
            return EmbeddingBasedChatContext(parent_chat=self)

        raise NotImplementedError(f"Unknown context model: {self.context_model}")

    @property
    def token_usage_db(self):
        """Return the chat's token usage database."""
        return TokenUsageDatabase(fpath=self.cache_dir / "chat_token_usage.db")

    @property
    def general_token_usage_db(self):
        """Return the general token usage database for all chats."""
        return TokenUsageDatabase(fpath=self.cache_dir.parent / "token_usage.db")

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
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def __del__(self):
        embedding_model = self.context_handler.database.get_embedding_model()
        chat_started = embedding_model is not None
        if self.private_mode or not chat_started:
            self.clear_cache()
        else:
            self.save_cache()

    @classmethod
    def from_dict(cls, configs: dict):
        """Creates a Chat instance from a configuration dictionary.

        Converts the configuration dictionary into a ChatOptions instance
        and uses it to instantiate the Chat class.

        Args:
            configs (dict): The chat configuration options as a dictionary.

        Returns:
            Chat: An instance of Chat initialized with the given configurations.
        """
        return cls(configs=ChatOptions.model_validate(configs))

    @classmethod
    def from_cli_args(cls, cli_args):
        """Creates a Chat instance from CLI arguments.

        Extracts relevant options from the CLI arguments and initializes a Chat instance
        with them.

        Args:
            cli_args: The command line arguments.

        Returns:
            Chat: An instance of Chat initialized with CLI-specified configurations.
        """
        chat_opts = {
            k: v
            for k, v in vars(cli_args).items()
            if k in ChatOptions.model_fields and v is not None
        }
        return cls.from_dict(chat_opts)

    @classmethod
    def from_cache(cls, cache_dir: Path):
        """Loads a chat instance from a cache directory.

        Args:
            cache_dir (Path): The path to the cache directory.

        Returns:
            Chat: An instance of Chat loaded with cached configurations and metadata.
        """
        try:
            with open(cache_dir / "configs.json", "r") as configs_f:
                new = cls.from_dict(json.load(configs_f))
            with open(cache_dir / "metadata.json", "r") as metadata_f:
                new.metadata = json.load(metadata_f)
                new.id = new.metadata["chat_id"]
        except FileNotFoundError:
            logger.warning(
                "Could not find configs and/or metadata file in cache directory."
                + "Creating chat with default configs."
            )
            new = cls()
        return new

    def load_history(self):
        """Load chat history from cache."""
        return self.context_handler.load_history()

    @property
    def initial_greeting(self):
        """Return the initial greeting for the chat."""
        return f"Hello! I'm {self.assistant_name}. How can I assist you today?"

    def respond_user_prompt(self, prompt: str, **kwargs):
        """Respond to a user prompt."""
        yield from self._respond_prompt(prompt=prompt, role="user", **kwargs)

    def respond_system_prompt(self, prompt: str, **kwargs):
        """Respond to a system prompt."""
        yield from self._respond_prompt(prompt=prompt, role="system", **kwargs)

    def yield_response_from_msg(self, prompt_msg: dict, add_to_history: bool = True):
        """Yield response from a prompt message."""
        # Get appropriate context for prompt from the context handler
        context = self.context_handler.get_context(msg=prompt_msg)

        # Make API request and yield response chunks
        full_reply_content = ""
        for chunk in make_api_chat_completion_call(
            conversation=[self.base_directive, *context, prompt_msg], chat_obj=self
        ):
            full_reply_content += chunk
            yield chunk

        if add_to_history:
            # Put current chat exchange in context handler's history
            self.context_handler.add_to_history(
                msg_list=[
                    prompt_msg,
                    {"role": "assistant", "content": full_reply_content},
                ]
            )

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
                    print(chunk, end="", flush=True)
                print()
                print()
        except (KeyboardInterrupt, EOFError):
            print("", end="\r")
            logger.info("Leaving chat.")
        except CannotConnectToApiError as error:
            print(f"{self.api_connection_error_msg}\n")
            logger.error("Leaving chat: {}", error)

    def start_talking(self):
        """Start the chat."""
        # ruff: noqa: T201
        from .text_to_speech import LiveAssistant

        lang = self.language_speech
        en_greeting = self.initial_greeting
        translation_prompt = f"Translate the greeting in the net line to {lang}. "
        translation_prompt += "Do NOT write anything else. Only the translation.\n"
        translation_prompt += f"{en_greeting}"
        initial_greeting = "".join(self.respond_system_prompt(prompt=translation_prompt))
        assistant = LiveAssistant(language=self.language_speech)
        assistant.speak(initial_greeting)
        try:
            previous_question_answered = True
            while True:
                if previous_question_answered:
                    logger.info(f"{self.assistant_name}> Listening...")
                question = assistant.listen()
                if not question:
                    previous_question_answered = False
                    continue
                logger.info(f"{self.assistant_name}> Let me think...")
                answer = "".join(self.respond_user_prompt(prompt=question))
                logger.info(f"{self.assistant_name}> Ok, here we go:")
                assistant.speak(answer)
                previous_question_answered = True
        except (KeyboardInterrupt, EOFError):
            print("", end="\r")
            logger.info("Leaving chat.")
        except CannotConnectToApiError as error:
            print(f"{self.api_connection_error_msg}\n")
            logger.error("Leaving chat: {}", error)

    def report_token_usage(self, report_current_chat=True, report_general: bool = False):
        """Report token usage and associated costs."""
        dfs = {}
        if report_general:
            dfs[
                "All Recorded Chats"
            ] = self.general_token_usage_db.get_usage_balance_dataframe()
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

    def _respond_prompt(self, prompt: str, role: str, **kwargs):
        prompt_as_msg = {"role": role.lower().strip(), "content": prompt.strip()}
        yield from self.yield_response_from_msg(prompt_as_msg, **kwargs)

    @property
    def api_connection_error_msg(self):
        """Return the error message for API connection errors."""
        return (
            "Sorry, I'm having trouble communicating with OpenAI right now. "
            + "Please check the validity of your API key and try again. "
            + "If the problem persists, please also take a look at the "
            + "OpenAI status page <https://status.openai.com>."
        )
