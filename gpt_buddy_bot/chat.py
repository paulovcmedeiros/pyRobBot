#!/usr/bin/env python3
from collections import defaultdict

import openai

from .chat_configs import ChatOptions
from .chat_context import BaseChatContext, EmbeddingBasedChatContext
from .tokens import TokenUsageDatabase, get_n_tokens


class Chat:
    def __init__(self, configs: ChatOptions):
        self.configs = configs
        for field in self.configs.model_fields:
            setattr(self, field, self.configs[field])

        self.token_usage = defaultdict(lambda: {"input": 0, "output": 0})
        self.token_usage_db = TokenUsageDatabase(fpath=self.token_usage_db_path)

        if self.context_model is None:
            self.context_handler = BaseChatContext(parent_chat=self)
        elif self.context_model == "text-embedding-ada-002":
            self.context_handler = EmbeddingBasedChatContext(
                embedding_model=self.context_model, parent_chat=self
            )
        else:
            raise NotImplementedError(f"Unknown context model: {self.context_model}")

    @property
    def base_directive(self):
        msg_content = " ".join(
            [
                instruction.strip()
                for instruction in [
                    f"Your name is {self.assistant_name}.",
                    f"You are a helpful assistant to {self.username}.",
                    "You answer correctly. You do not lie.",
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
    def from_cli_args(cls, cli_args):
        configs = ChatOptions.model_validate(vars(cli_args))
        return cls(configs)

    def respond_user_prompt(self, prompt: str):
        yield from self._respond_prompt(prompt=prompt, role="user")

    def respond_system_prompt(self, prompt: str):
        yield from self._respond_prompt(prompt=prompt, role="system")

    def yield_response_from_msg(self, prompt_as_msg: dict):
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
        for chunk in _make_api_call(conversation=contextualised_prompt, model=self.model):
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
        try:
            while True:
                question = input(f"{self.username}: ").strip()
                if not question:
                    continue
                print(f"{self.assistant_name}: ", end="", flush=True)
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


def _make_api_call(conversation: list, model: str):
    success = False
    while not success:
        try:
            for line in openai.ChatCompletion.create(
                model=model,
                messages=conversation,
                request_timeout=10,
                stream=True,
                temperature=0.8,
            ):
                reply_chunk = getattr(line.choices[0].delta, "content", "")
                yield reply_chunk
        except (
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
        ) as error:
            print(f"    > {error}. Retrying...")
        else:
            success = True
