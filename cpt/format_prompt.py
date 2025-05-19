from datasets import load_dataset, Dataset, DatasetDict
import random

def formatting_prompts_func(examples, tokenizer, eval):
    EOS_TOKEN = tokenizer.eos_token
    questions    = examples["question"]
    choices      = examples["choices"]
    answers      = examples['answer']
    outputs = []
    for question, choice, answer in zip(questions, choices, answers):
        input = f"{question}\nA. {choice[0]}\nB. {choice[1]}\nC. {choice[2]}\nD. {choice[3]}"
        text = eval.cpt_prompt.format(input, answer) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }

def is_english_and_long(example):
    text = example["text"]
    return text.startswith("[EN]") and len(text.split()) > 50  # 50 words as a rough paragraph threshold

def is_english_and_long(example):
    text = example["text"]
    return text.startswith("[EN]") and len(text.split()) > 40  # 40 words as a rough paragraph threshold

def add_data_noise(dataset):
    outputs = [example["text"] for example in dataset['train']]
    data = load_dataset("dehanalkautsar/WikiMultiV2_partition", split='test')
    filtered_data = data.filter(is_english_and_long)
    filtered_subset = filtered_data.select(range(min(600, len(filtered_data))))
    outputs.extend(filtered_subset['text'])
    random.shuffle(outputs)
    noisy_train_dataset = Dataset.from_dict({"text": outputs})
    noisy_dataset_dict = DatasetDict({
        'train': noisy_train_dataset
    })
    return noisy_dataset_dict

def tokenize_function(examples, tokenizer):
    tokens = tokenizer(examples["text"], truncation=True, padding="longest", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # For causal LM, labels = input_ids
    return tokens