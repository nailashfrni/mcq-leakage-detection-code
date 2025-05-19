import json
import argparse
import pandas as pd
import os

from evals.hellaswag_eval import HellaSwagEval
from evals.mmlu_eval import MMLUEval

NUM_INSTANCES = 600
TRAIN_INSTANCES = 300

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='run_cpt', description='Filter Dataset')
    parser.add_argument("--df_file", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--eval", type=str, default='mmlu')
    args = parser.parse_args()


    if args.eval == 'mmlu':
        eval = MMLUEval(args.df_file)
    else:
        eval = HellaSwagEval(args.df_file)
    
    avg_ppl = eval.df[eval.df['Score'] == 0]['Perplexity'].mean()
    try:
        print('Average PPL:', avg_ppl)
        temp_df = eval.df[(eval.df['Score'] == 0.0) & (eval.df['Perplexity'] > avg_ppl)]
        sampled_df = temp_df.sample(n=NUM_INSTANCES, random_state=42)
    except:
        try:
            median_ppl = eval.df[eval.df['Score'] == 0]['Perplexity'].median()
            print('Median PPL:', median_ppl)
            temp_df = eval.df[(eval.df['Score'] == 0.0) & (eval.df['Perplexity'] > median_ppl)]
            sampled_df = temp_df.sample(n=NUM_INSTANCES, random_state=42)
        except:
            temp_df = eval.df[(eval.df['Score'] == 0.0) & (eval.df['Perplexity'] > 100)]
            sampled_df = temp_df.sample(n=NUM_INSTANCES, random_state=42)

    sampled_df.reset_index(drop=True, inplace=True)
    sampled_df = sampled_df.fillna('')
    df_train = sampled_df.sample(n=TRAIN_INSTANCES, random_state=42)
    df_test = sampled_df.drop(df_train.index)

    train_data = [
        {
            "id": eval.get_id(row),
            "question": eval.get_row_question(row),
            "choices": eval.get_row_options(row),
            "answer": eval.get_row_answer(row)
        }
        for _, row in df_train.iterrows()
    ]
    model_title = args.model_dir.split('/')[-1]
    os.makedirs(f"{args.prefix}/cpt/data/{args.eval}/", exist_ok=True)
    with open(f"{args.prefix}/cpt/data/{args.eval}/cpt_dataset_{args.eval}_{model_title}_train.json", "w") as f:
        json.dump(train_data, f, indent=4)
    print(f"Dataset saved as {args.prefix}/cpt/data/{args.eval}/cpt_dataset_{args.eval}_{model_title}_train.json")

    test_data = [
        {
            "id": eval.get_id(row),
            "question": eval.get_row_question(row),
            "choices": eval.get_row_options(row),
            "answer": eval.get_row_answer(row)
        }
        for _, row in df_test.iterrows()
    ]

    for d in train_data:
        d['gold_leakage'] = 1
    for d in test_data:
        d['gold_leakage'] = 0

    results = train_data + test_data
    with open(f"{args.prefix}/data/{args.eval}/cpt_dataset_{args.eval}_{model_title}.json", 'w') as f:
        json.dump(results, f, indent=4)
        print(f'Save data to {args.prefix}/data/{args.eval}/cpt_dataset_{args.eval}_{model_title}.json')