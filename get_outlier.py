from sklearn.ensemble import IsolationForest
import numpy as np
import json
import argparse
import os
import tqdm

from evals.hellaswag_eval import HellaSwagEval
from evals.mmlu_eval import MMLUEval

'''
Code is adapted from https://github.com/nishiwen1214/Benchmark-leakage-detection
'''

def display_matching_accuracy(list1, list2):
    assert len(list1) == len(list2), "Lists must have the same length"
    matches = sum(a == b for a, b in zip(list1, list2))
    print(f'Matching Accuracy: {matches / len(list1)} ({matches}/{len(list1)})')
    return matches, matches / len(list1)

def matching_counter(list1, list2, is_leakage):
    assert len(list1) == len(list2), "Lists must have the same length"
    symbol = 1 if is_leakage else 0
    matches = sum(a == b == symbol for a, b in zip(list1, list2))
    return matches

def hamming_distance(list1, list2):
    assert len(list1) == len(list2), "Lists must have the same length"
    count = 0
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            count += 1
    return count

parser = argparse.ArgumentParser(prog='get_outlier', description='')
parser.add_argument("--logprobs_dir", type=str)
parser.add_argument("--permutations_data_dir", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--prefix", type=str)
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--eval", type=str, default='mmlu')
parser.add_argument("--model_name", type=str)
parser.add_argument("--is_selected", type=bool, default=False)
parser.add_argument("--permutation_num", type=int)
parser.add_argument(
        "--groups", 
        nargs="*",
        type=str,
        help="Select specific list of groups (optional)"
    )
parser.add_argument(
        "--subjects", 
        nargs="*",
        type=str,
        help="Select specific list of subjects (optional)"
    )
args = parser.parse_args()
thresholds = [-0.2, -0.17, -0.15]

model_title = args.model_name.split('/')[-1]
with open(args.permutations_data_dir, 'r', encoding='utf8') as file:
    list_data = json.load(file)
with open(args.logprobs_dir, 'r', encoding='utf8') as file:
    list_logprobs = json.load(file)

with open(f'{args.prefix}/data/{args.eval}/cpt_dataset_{args.eval}_{model_title}.json', 'r', encoding='utf8') as file:
    dataset = json.load(file)

if args.eval == 'mmlu':
    eval = MMLUEval()
else:
    eval = HellaSwagEval()


list_data = [list_data[i:i + args.permutation_num] for i in range(0, len(list_data), args.permutation_num)]
list_logprobs = [list_logprobs[i:i + args.permutation_num] for i in range(0, len(list_logprobs), args.permutation_num)]
y_true = [d['gold_leakage'] for d in dataset]

# For permutation-R with p=50%, filter each chunk to only keep selected_perm indices
if args.is_selected:
    selected_perm = [0, 1, 2, 6, 9, 10, 12, 13, 18, 19, 20, 22]
    list_data = [[chunk[i] for i in selected_perm] for chunk in list_data]
    list_logprobs = [[chunk[i] for i in selected_perm] for chunk in list_logprobs]

selected_prefix = 'new_selected_' if args.is_selected else ''
if 'quad' in args.permutations_data_dir:
    selected_prefix = 'quad_'
batch_suffix = f'_part_{args.batch}' if args.batch > 0 else ''

checkpoint_epoch = 0
if 'cp' in args.logprobs_dir:
    checkpoint_epoch = int(args.logprobs_dir.split("-")[-1].replace(".json", ""))
cp_epoch_suffix = f"_cp-epoch-{checkpoint_epoch}" if (checkpoint_epoch > 0) else ""

base_dir = f'{args.prefix}/permutation/result/{model_title}'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(base_dir + f'/{args.eval}', exist_ok=True)
os.makedirs(base_dir + f'/{args.eval}/leakages', exist_ok=True)
os.makedirs(base_dir + f'/{args.eval}/outliers', exist_ok=True)

if args.method == "shuffled":
    leakage_info = [[], [], []]
    outliers = [[], [], []]
    y_pred = [[], [], []]
    for index, data in enumerate(tqdm.tqdm(list_logprobs)):
        X = np.array(data).reshape(-1, 1)
        clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        clf.fit(X)
        scores = clf.decision_function(X)
        max_value_index = np.argmax(data)
        max_value_score = scores[max_value_index]
        for outlier_index, threshold in enumerate(thresholds):
            leakage = 0
            if max_value_score < threshold:
                outlier = {
                    # "index": str(index),
                    "max_value_index": str(max_value_index),
                    "id": eval.get_id(list_data[index][max_value_index]),
                    "data": list_data[index][max_value_index]["instruction"],
                    "logprobs": data[max_value_index],
                    "gold_leakage": dataset[index]['gold_leakage']
                }
                outliers[outlier_index].append(outlier)
                leakage = 1
            leakage_info[outlier_index].append({
                'id': eval.get_id(list_data[index][max_value_index]),
                'leakage': leakage,
                "gold_leakage": dataset[index]['gold_leakage']
            })
            y_pred[outlier_index].append(leakage)

    for i, threshold in enumerate(thresholds):
        with open(base_dir + f'/{args.eval}/outliers/{selected_prefix}outliers{threshold}{batch_suffix}{cp_epoch_suffix}.json', 'w', encoding='utf8') as json_file:
            json.dump(outliers[i], json_file, indent=4, ensure_ascii=False)
            print(f'Save result to {base_dir}/{args.eval}/outliers/{selected_prefix}outliers{threshold}{batch_suffix}{cp_epoch_suffix}.json')
        with open(base_dir + f'/{args.eval}/leakages/{selected_prefix}leakage{threshold}{batch_suffix}{cp_epoch_suffix}.json', 'w', encoding='utf8') as json_file:
            json.dump(leakage_info[i], json_file, indent=4, ensure_ascii=False)
            print(f'Save result to {base_dir}/{args.eval}/leakages/{selected_prefix}leakage{threshold}{batch_suffix}{cp_epoch_suffix}.json')
else:
    leakage_info = []
    outliers = []
    y_pred = []
    for index, data in enumerate(list_logprobs):
        leakage = 0
        max = data[0]
        isMax = True
        for temp in data[1:]:
            if temp > max:
                isMax = False
                break
        if isMax:
            dict = {
                # "index": str(index),
                "max_value_index": str(0),
                "id": eval.get_id(list_data[index][0]),
                "data": list_data[index][0]["instruction"],
                "logprobs": data[0],
                "gold_leakage": dataset[index]['gold_leakage']
            }
            outliers.append(dict)
            leakage = 1
        leakage_info.append({
            'id': eval.get_id(list_data[index][0]),
            'leakage': leakage,
            "gold_leakage": dataset[index]['gold_leakage']
        })
        y_pred.append(leakage)

    with open(base_dir + f'/{args.eval}/outliers/{selected_prefix}outliers_max{batch_suffix}{cp_epoch_suffix}.json', 'w', encoding='utf8') as json_file:
        json.dump(outliers, json_file, indent=4, ensure_ascii=False)
        print(f'Save result to {base_dir}/{args.eval}/outliers/{selected_prefix}outliers_max{batch_suffix}{cp_epoch_suffix}.json')
    with open(base_dir + f'/{args.eval}/leakages/{selected_prefix}leakage_max{batch_suffix}{cp_epoch_suffix}.json', 'w', encoding='utf8') as json_file:
        json.dump(leakage_info, json_file, indent=4, ensure_ascii=False)
        print(f'Save result to {base_dir}/{args.eval}/leakages/{selected_prefix}leakage_max{batch_suffix}{cp_epoch_suffix}.json')
        