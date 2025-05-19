from model_callers.model_caller import ModelCaller
from dotenv import load_dotenv
import os
import time

import openai
from openai import OpenAI

class GPTModelCaller(ModelCaller):
    def __init__(self, 
                model_name: str = "gpt-3.5-turbo",
                temperature: float = 0.7):
        super().__init__(model_name)
        load_dotenv()
        self.api_key_name = "OPENAI_API_KEY"
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()
    
    def __call__(self, message: str, temperature: float, max_tokens: int):
        if self.system_message:
            message_list = [self.pack_message("system", self.system_message),
                            self.pack_message("user", message)]
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1