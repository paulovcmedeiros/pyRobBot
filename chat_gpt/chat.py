#!/usr/bin/env python3
import openai

from . import GeneralConstants
from .chat_context import BaseChatContext, EmbeddingBasedChatContext
from .tokens import TokenUsageDatabase


class Chat:
    def __init__(
        self, model: str, base_instructions: str, send_full_history: bool = False
    ):
        self.model = model
        self.username = "chat_user"
        self.assistant_name = f"chat_{model.replace('.', '_')}"
        self.system_name = "chat_manager"

        self.ground_ai_instructions = " ".join(
            [
                instruction.strip()
                for instruction in [
                    f"Your name is {self.assistant_name}",
                    f"You are a helpful assistant to {self.username}.",
                    "You answer correctly. You do not lie.",
                    f"{base_instructions.strip(' .')}.",
                    f"You follow all directives by {self.system_name}.",
                ]
                if instruction.strip()
            ]
        )

        self.token_usage = {"input": 0, "output": 0}
        self.token_usage_db = TokenUsageDatabase(
            fpath=GeneralConstants.TOKEN_USAGE_DATABASE,
            model=self.model,
        )

        if send_full_history:
            self.context_handler = BaseChatContext(parent_chat=self)
        else:
            self.context_handler = EmbeddingBasedChatContext(parent_chat=self)

        self.query_context = [
            {
                "role": "system",
                "name": self.system_name,
                "content": self.ground_ai_instructions,
            }
        ]

    def __del__(self):
        # Store token usage to database
        self.token_usage_db.insert_data(
            n_input_tokens=self.token_usage["input"],
            n_output_tokens=self.token_usage["output"],
        )
        self.report_token_usage()

    def yield_response(self, question: str):
        question = question.strip()

        # Add context to the conversation
        self.query_context = self.context_handler.add_user_input(
            conversation=self.query_context, user_input=question
        )

        # Update number of input tokens
        self.token_usage["input"] += sum(
            self.token_usage_db.get_n_tokens(string=msg["content"])
            for msg in self.query_context
        )

        full_reply_content = ""
        for chunk in _make_api_call(conversation=self.query_context, model=self.model):
            full_reply_content += chunk
            yield chunk

        # Update number of output tokens
        self.token_usage["output"] += self.token_usage_db.get_n_tokens(full_reply_content)

        # Update context with the reply
        self.query_context = self.context_handler.add_chat_reply(
            conversation=self.query_context, chat_reply=full_reply_content.strip()
        )

    def start(self):
        try:
            while True:
                question = input(f"{self.username}: ").strip()
                if not question:
                    continue
                print(f"{self.assistant_name}: ", end="", flush=True)
                for chunk in self.yield_response(question=question):
                    print(chunk, end="", flush=True)
                print()
                print()
        except (KeyboardInterrupt, EOFError):
            print("Exiting chat.")

    def report_token_usage(self):
        self.token_usage_db.print_usage_costs(self.token_usage)


def _make_api_call(conversation: list, model: str):
    success = False
    while not success:
        try:
            for line in openai.ChatCompletion.create(
                model=model,
                messages=conversation,
                request_timeout=30,
                stream=True,
                temperature=0.8,
            ):
                reply_content_token = getattr(line.choices[0].delta, "content", "")
                yield reply_content_token
                success = True
        except (
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
        ) as error:
            print(f"    > {error}. Retrying...")
