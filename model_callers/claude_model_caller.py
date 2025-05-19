import time
from dotenv import load_dotenv
import os
import anthropic
from model_callers.model_caller import ModelCaller

CLAUDE_SYSTEM_MESSAGE_LMSYS = (
    "You are a helpful assistant."
)

class ClaudeModelCaller(ModelCaller):
    """
    Caller for Gemini Model API
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022"
    ):
        super().__init__(model_name)
        load_dotenv()
        self.api_key_name = "ANTHROPIC_API_KEY"
        self.api_key=os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.system_message = CLAUDE_SYSTEM_MESSAGE_LMSYS
        self.image_format = "base64"

    def __call__(self, message: str, temperature: float, max_tokens: int):
        if self.system_message:
            message_list = [self.pack_message("user", message)]
        trial = 0
        while True:
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    system=self.system_message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=message_list,
                )
                return message.content[0].text
            except anthropic.RateLimitError as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            except Exception as e:
                exception_backoff = 2**trial
                print('Error exception', e)
                time.sleep(exception_backoff)
                trial += 1