from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from cpt.callback import SaveEveryEpochCallback
from cpt.custom_trainer import CustomPeftTrainer
from cpt.format_prompt import *

from evals.hellaswag_eval import HellaSwagEval
from evals.mmlu_eval import MMLUEval

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Define LoRA configuration for continual pretraining
    lora_config = LoraConfig(
        r=128,  # LoRA rank
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def load_tokenized_dataset(data_files, tokenizer, eval, add_noise=False):
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer, eval), batched = True,)
    dataset = dataset.select_columns(["text"])

    if add_noise:
        dataset = add_data_noise(dataset)

    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    return tokenized_dataset

def train(model_name, model, tokenizer, tokenized_dataset, 
            num_epochs, hf_repo):
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        learning_rate = 1e-3,
        lr_scheduler_type = "cosine",
        weight_decay=0.01,
        logging_strategy="epoch",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Optional: change to "wandb" or "tensorboard" for logging
        save_total_limit=1,
        save_strategy="epoch",
        push_to_hub=False,
        optim="adamw_8bit",
        fp16=True
    )
    trainer = CustomPeftTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        tokenizer=tokenizer,
        callbacks=[SaveEveryEpochCallback(hf_repo)]
    )
    trainer_stats = trainer.train()
    return trainer, trainer_stats

def save_loss_graph(history, model_title, output_file):
    plt.figure(figsize=(10, 4))
    plt.plot(history["epoch"], history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Continual Pre-Training Loss {model_title} Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(output_file + '.pdf', format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='run_cpt', description='Continual Pre-Training on Transformer Models')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_files", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--hf_repo", type=str)
    args = parser.parse_args()

    model_title = args.model_name.split('/')[-1]
    output_file = f"cpt/loss_cpt_{model_title}_{args.eval}"

    eval = None
    if args.eval == 'mmlu':
        eval = MMLUEval()
    else:
        eval = HellaSwagEval()

    model, tokenizer = load_model(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = load_tokenized_dataset(args.data_files, tokenizer, eval, args.add_noise)
    trainer, trainer_stats = train(args.model_name, 
                                   model, tokenizer, tokenized_dataset, 
                                   args.num_epochs, args.hf_repo)
    
    history = pd.DataFrame(trainer.state.log_history)
    history['epoch'] = history['epoch'].apply(lambda x: int(x))

    save_loss_graph(history, model_title, output_file)
    history.to_csv(output_file + '.csv', index=False)