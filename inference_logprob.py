import torch
import torch.nn.functional as F
import json
import tqdm
import os
import argparse
from utils import load_model
import time

'''
Code is adapted from https://github.com/nishiwen1214/Benchmark-leakage-detection
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='inference logprob', description='Inference logprob from LLM')
    parser.add_argument("--base_model_dir", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--permutations_data_dir", type=str)
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--fine_tune_type", type=str, default=None)
    parser.add_argument("--eval", type=str, default='mmlu')
    parser.add_argument("--checkpoint_epoch", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model_dir, args.adapter_dir,
                                  args.checkpoint_epoch, args.fine_tune_type)

    def find_indices(lst, value):
        indices = []
        for i, elem in enumerate(lst):
            if (elem == value and len(lst[i + 1]) != 0 and lst[i + 1][0] == ".") or elem == 'A.':
                indices.append(i)
                return indices
        return indices


    def score(prompt):
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
            input_tokens = [tokenizer.decode([id]) for id in input_ids[0]]
            index = find_indices(input_tokens, 'A')
            logits = model(input_ids).logits
            all_tokens_logprobs = F.log_softmax(logits.double(), dim=2)
            input_logprobs = [all_tokens_logprobs[:, k - 1, input_ids[0, k]] for k in range(1, input_ids.shape[1])]
            input_logprobs = [input_logprobs[k].detach().cpu().numpy()[0] for k in range(len(input_logprobs))]
            del logits
            return input_tokens, input_logprobs, index[0]


    def display(prompt):
        _, input_logprobs, index = score(prompt)
        all_logprobs = 0
        for i in range(index, len(input_logprobs)):
            all_logprobs = all_logprobs + input_logprobs[i]
        return all_logprobs


    with open(args.permutations_data_dir, 'r', encoding='utf8') as file:
        datas = json.load(file)

    cp_epoch_suffix = f"_{args.fine_tune_type}-cp-epoch-{args.checkpoint_epoch}" if (args.fine_tune_type is not None and args.checkpoint_epoch > 0) else ""
    batch_suffix = f'_part_{args.batch}' if args.batch > 0 else ''

    logprobs_list = []
    start_time = time.time()
    for _,data in enumerate(tqdm.tqdm(datas)):
        result = display(data["instruction"])
        logprobs_list.append(result)
        torch.cuda.empty_cache()

    model_title = args.base_model_dir.split('/')[-1]

    base_dir = f'{args.prefix}/permutation/result/{model_title}'
        
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(base_dir + f'/{args.eval}'):
        os.mkdir(base_dir + f'/{args.eval}')
    if not os.path.exists(base_dir + f'/{args.eval}/logprobs'):
        os.mkdir(base_dir + f'/{args.eval}/logprobs')

    fileprefix = ''
    if 'quad' in args.permutations_data_dir:
        fileprefix = 'quad_'
    output_file = base_dir + f"/{args.eval}/logprobs/{fileprefix}logprobs_{model_title}_{args.eval}{batch_suffix}{cp_epoch_suffix}.json"

    with open(output_file, 'w', encoding='utf8') as json_file:
        json.dump(logprobs_list, json_file, indent=4, ensure_ascii=False)
    print(f'Save result to {output_file}')