import time
from dotenv import load_dotenv
import os
import google.generativeai as genai
from model_callers.model_caller import ModelCaller

class GeminiModelCaller(ModelCaller):
    """
    Caller for Gemini Model API
    """

    def __init__(self, model_name):
        super().__init__(model_name)
        load_dotenv()
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel(model_name)

    def __call__(self, message: str, temperature: float, max_tokens: int):
        trial = 0
        while True:
            try:
                response = self.model.generate_content(message,
                                                       generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                })
                if not response.candidates:
                    feedback = getattr(response, "prompt_feedback", None)
                    try:
                        if feedback is not None:
                            block_reason = getattr(feedback, "block_reason", '')
                            return block_reason.name
                    except:
                        return 'No candidates -- possibly blocked'
                else:
                    candidate = response.candidates[0]
                    if candidate.finish_reason.name == "SAFETY":
                        return "Response blocked due to safety filters (finish_reason = SAFETY)"
                    elif not candidate.content.parts:
                        return "Candidate returned no valid parts."
                return response.text
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1