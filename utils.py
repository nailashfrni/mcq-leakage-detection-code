import torch
import openai
from peft import PeftModel
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_callers.gpt_model_caller import GPTModelCaller
from model_callers.gemini_model_caller import GeminiModelCaller
from model_callers.claude_model_caller import ClaudeModelCaller

def load_model(base_model_dir, adapter_dir, checkpoint_epoch, fine_tune_type):
    """
    Load either OpenAI API or Hugging Face model.
    """
    print(f'Load model {base_model_dir}...')

    model_callers = {
        'gpt': GPTModelCaller(base_model_dir),
        'gemini': GeminiModelCaller(base_model_dir),
        'claude': ClaudeModelCaller(base_model_dir)
    }
    model_type = base_model_dir.split('-')[0]
    if model_type in model_callers.keys():
        return model_callers[model_type], None
    
    if fine_tune_type not in ['ift', 'cpt']:
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        model = AutoModelForCausalLM.from_pretrained(base_model_dir, device_map="auto",
                                                    torch_dtype=torch.float16, 
                                                     )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True,
                                                    torch_dtype=torch.float16, device_map="auto")
        peft_model = PeftModel.from_pretrained(
                        base_model,
                        adapter_dir,
                        revision=f"checkpoint-epoch-{checkpoint_epoch}"
                    )
        model = peft_model.merge_and_unload()
    return model, tokenizer

def pack_message(role, content):
    return {'role': role, 'content': content}

def generate_answer(model, prompt, model_name):
    message_list = [pack_message("system", "You are a helpful assistant."),
                    pack_message("user", prompt)]
    trial = 0
    while True:
        try:
            response = model.chat.completions.create(
                model=model_name,
                messages=message_list
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