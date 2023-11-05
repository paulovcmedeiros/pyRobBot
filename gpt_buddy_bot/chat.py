#!/usr/bin/env python3
import uuid
from collections import defaultdict

import openai

from . import GeneralConstants
from .chat_configs import ChatOptions, OpenAiApiCallOptions
from .chat_context import BaseChatContext, EmbeddingBasedChatContext
from .tokens import TokenUsageDatabase, get_n_tokens


class Chat:
    def __init__(self, configs: ChatOptions):
        self.id = uuid.uuid4()

        self._passed_configs = configs
        for field in self._passed_configs.model_fields:
            setattr(self, field, self._passed_configs[field])

        self.token_usage = defaultdict(lambda: {"input": 0, "output": 0})
        self.token_usage_db = TokenUsageDatabase(fpath=self.token_usage_db_path)

        if self.context_file_path is None:
            self.context_file_path = (
                GeneralConstants.PACKAGE_TMPDIR / f"embeddings_for_chat_{self.id}.csv"
            )

        if self.context_model is None:
            self.context_handler = BaseChatContext(parent_chat=self)
        elif self.context_model == "text-embedding-ada-002":
            self.context_handler = EmbeddingBasedChatContext(parent_chat=self)
        else:
            raise NotImplementedError(f"Unknown context model: {self.context_model}")

    @property
    def configs(self):
        """Return the chat's configs after initialisation."""
        configs_dict = {}
        for field_name in ChatOptions.model_fields:
            configs_dict[field_name] = getattr(self, field_name)
        return ChatOptions.model_validate(configs_dict)

    @property
    def base_directive(self):
        msg_content = " ".join(
            [
                instruction.strip()
                for instruction in [
                    f"You are {self.assistant_name} (model {self.model}).",
                    f"You are a helpful assistant to {self.username}.",
                    " ".join(
                        [f"{instruct.strip(' .')}." for instruct in self.ai_instructions]
                    ),
                    f"You must remember and follow all directives by {self.system_name}.",
                ]
                if instruction.strip()
            ]
        )
        return {"role": "system", "name": self.system_name, "content": msg_content}

    def __del__(self):
        # Store token usage to database
        for model in [self.model, self.context_model]:
            self.token_usage_db.insert_data(
                model=model,
                n_input_tokens=self.token_usage[model]["input"],
                n_output_tokens=self.token_usage[model]["output"],
            )
        if self.report_accounting_when_done:
            self.report_token_usage()

    @classmethod
    def from_dict(cls, configs: dict):
        return cls(configs=ChatOptions.model_validate(configs))

    @classmethod
    def from_cli_args(cls, cli_args):
        chat_opts = {
            k: v
            for k, v in vars(cli_args).items()
            if k in ChatOptions.model_fields and v is not None
        }
        return cls.from_dict(chat_opts)

    @property
    def initial_greeting(self):
        return f"Hello! I'm {self.assistant_name}. How can I assist you today?"

    def respond_user_prompt(self, prompt: str):
        yield from self._respond_prompt(prompt=prompt, role="user")

    def respond_system_prompt(self, prompt: str):
        yield from self._respond_prompt(prompt=prompt, role="system")

    def yield_response_from_msg(self, prompt_as_msg: dict):
        """Yield response from a prompt."""
        try:
            yield from self._yield_response_from_msg(prompt_as_msg=prompt_as_msg)
        except openai.error.AuthenticationError:
            yield "Sorry, I'm having trouble authenticating with OpenAI. "
            yield "Please check the validity of your API key and try again."

    def _yield_response_from_msg(self, prompt_as_msg: dict):
        """Yield response from a prompt. Assumes that OpenAI authentication works."""
        role = prompt_as_msg["role"]
        prompt = prompt_as_msg["content"]

        # Get appropriate context for prompt from the context handler
        prompt_context_request = self.context_handler.get_context(text=prompt)
        context = prompt_context_request["context_messages"]

        # Update token_usage with tokens used in context handler for prompt
        self.token_usage[self.context_model]["input"] += sum(
            prompt_context_request["tokens_usage"].values()
        )

        contextualised_prompt = [self.base_directive, *context, prompt_as_msg]
        # Update token_usage with tokens used in chat input
        self.token_usage[self.model]["input"] += sum(
            get_n_tokens(string=msg["content"], model=self.model)
            for msg in contextualised_prompt
        )

        # Make API request and yield response chunks
        full_reply_content = ""
        for chunk in _make_api_chat_completion_call(
            conversation=contextualised_prompt, chat_obj=self
        ):
            full_reply_content += chunk
            yield chunk

        # Update token_usage ith tokens used in chat output
        self.token_usage[self.model]["output"] += get_n_tokens(
            string=full_reply_content, model=self.model
        )

        # Put current chat exchange in context handler's history
        history_entry_registration_tokens_usage = self.context_handler.add_to_history(
            text=f"{role}: {prompt}. {self.assistant_name}: {full_reply_content}"
        )

        # Update token_usage with tokens used in context handler for reply
        self.token_usage[self.context_model]["output"] += sum(
            history_entry_registration_tokens_usage.values()
        )

    def start(self):
        """Start the chat."""
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
            print("Exiting chat.")

    def report_token_usage(self, current_chat: bool = True):
        self.token_usage_db.print_usage_costs(self.token_usage, current_chat=current_chat)

    def _respond_prompt(self, prompt: str, role: str):
        prompt = prompt.strip()
        role = role.lower().strip()
        role2name = {"user": self.username, "system": self.system_name}
        prompt_as_msg = {"role": role, "name": role2name[role], "content": prompt}
        yield from self.yield_response_from_msg(prompt_as_msg)


def _make_api_chat_completion_call(conversation: list, chat_obj: Chat):
    success = False

    api_call_args = {}
    for field in OpenAiApiCallOptions.model_fields:
        if getattr(chat_obj, field) is not None:
            api_call_args[field] = getattr(chat_obj, field)

    while not success:
        try:
            for line in openai.ChatCompletion.create(
                messages=conversation,
                stream=True,
                **api_call_args,
            ):
                reply_chunk = getattr(line.choices[0].delta, "content", "")
                yield reply_chunk
        except (
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
        ) as error:
            print(f"\n    > {error}. Retrying...")
        else:
            success = True
