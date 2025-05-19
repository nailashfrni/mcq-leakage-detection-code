import json
import itertools
import argparse
import os
import math

from evals.hellaswag_eval import HellaSwagEval
from evals.mmlu_eval import MMLUEval

'''
Code is adapted from https://github.com/nishiwen1214/Benchmark-leakage-detection
{
    'question': 'Which of the following descriptions of branchial arches is incorrect'
    'choices': ['Formed by mesenchymal proliferation',
        'Appears in the 4th week of human embryo',
        'Branial grooves are between adjacent branchial arches',
        'There are 5 pairs of branchial arches in total']
}

Cmd example:
python data_process.py  --data_dir data/mmlu/mmlu.csv --eval mmlu --save_dir permutation/data/mmlu --num_batches 5
'''
parser = argparse.ArgumentParser(prog='data_process', description='')
parser.add_argument("--data_dir", type=str)
parser.add_argument("--eval", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--num_batches", type=int, default=1)
parser.add_argument("--fine_tune_type", type=str, default=None) # either ift or cpt
args = parser.parse_args()

chars = [0, 1, 2, 3]   # ['A', 'B', 'C', 'D']
if args.eval == 'mmlu':
    eval = MMLUEval(args.data_dir)
else:
    eval = HellaSwagEval(args.data_dir)

def get_questions(row, perm):
    question = eval.get_row_question(row)
    options = eval.get_row_options(row)
    return f'''{question}:
A. {options[perm[0]]}
B. {options[perm[1]]}
C. {options[perm[2]]} 
D. {options[perm[3]]}'''

def prompt_ift(row, perm):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following multiple choice question.

### Input:
{get_questions(row, perm)}

### Response:
"""

def prompt_cpt(row, perm):
    return f"""Answer the following multiple choice question.
### Question: {get_questions(row, perm)}

### Answer: """

def prompt_original(row, perm):
    return f"""{get_questions(row, perm)}"""

# Define options
permutations_list = list(itertools.permutations(chars))

num_batches = args.num_batches   # 1
instance_per_batch = math.ceil(eval.df.shape[0] / num_batches)
for batch_idx in range(num_batches):
    start_idx = batch_idx * instance_per_batch
    end_idx = (batch_idx + 1) * instance_per_batch
    batch_df = eval.df.iloc[start_idx:end_idx]

    result = []

    for index, row in batch_df.iterrows(): 

        for perm in permutations_list:
            instruction = {
                "id": row['id'],
                "instruction": prompt_original(row, perm) if args.fine_tune_type is None
                                else prompt_cpt(row, perm)
            }
            result.append(instruction)

    os.makedirs(args.save_dir, exist_ok=True)
    filename = args.data_dir.split('/')[-1].replace('_' + args.eval, '').replace('.csv', '').replace('.json', '')
    batch_suffix = f'_{batch_idx + 1}' if args.num_batches > 1 else ''
    final_filename = f"{args.save_dir}/permutations_data_{filename}{batch_suffix}.json"
    with open(final_filename, 'w', encoding='utf8') as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)
    print('Save data to', final_filename)