import argparse
import os

from utils import *
from sklearn.metrics import precision_score, recall_score, accuracy_score

from model_callers.model_caller import ModelCaller
from evals.hellaswag_eval import HellaSwagEval
from evals.mmlu_eval import MMLUEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Benchmark Leakage Detection based on N-Gram Accuracy', add_help=False)
    parser.add_argument('--input_file', type=str, required=True, help='path to dataset')
    parser.add_argument('--base_model_dir', type=str, required=True, help='model name (Hugging Face or OpenAI)')
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--epoch', type=int, help='n-gram', default=0)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--method', type=str, default=0)
    parser.add_argument('--eval', type=str, default='mmlu')
    parser.add_argument('--prefix', type=str, default='.')
    parser.add_argument('--fine_tune_type', type=str, default=None)

    # Semihalf additional args
    parser.add_argument('--mode', type=str)
    parser.add_argument('--output_type', type=str, default='json')
    parser.add_argument('--compute_ppl', type=bool, default=False)

    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model_dir, args.adapter_dir,
                                args.epoch, args.fine_tune_type)

    is_close_weight = True if isinstance(model, ModelCaller) else False
    eval = None
    if args.eval == 'mmlu':
        eval = MMLUEval(args.input_file, is_close_weight)
    else:
        eval = HellaSwagEval(args.input_file, is_close_weight)

    fine_tune_suffix = f'_{args.fine_tune_type}_cp{args.epoch}' if args.fine_tune_type is not None else ''
    model_title = args.base_model_dir.split('/')[-1]
    batch_suffix = f'_part_{args.batch}' if args.batch > 0 else ''

    base_dir = f'{args.prefix}/{args.method}/result/{model_title}'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(base_dir + f'/{args.eval}', exist_ok=True)

    method = None
    match args.method:
        case "semihalf":
            from methods.semihalf_method import SemihalfMethod
            method = SemihalfMethod(eval, args.prefix, model, tokenizer, args.mode, args.output_type, 
                args.compute_ppl, args.batch)
            output_file = base_dir + f'/{args.eval}/{args.mode}_{args.eval}{batch_suffix}_{model_title}{fine_tune_suffix}'
        case "ngram":
            from methods.ngram_method import NgramMethod
            method = NgramMethod(eval, args.prefix, model, tokenizer, args.base_model_dir, args.batch)
            output_file = base_dir + f'/{args.eval}/ngram_{args.eval}{batch_suffix}_{model_title}{fine_tune_suffix}.json'

    print(f'{args.method.capitalize()} Evaluation on {args.eval} CPT Epoch {args.epoch}')
    y_true, y_pred = method.evaluate(output_file)

    if y_true is not None and y_pred is not None:
        print(f'Recall: {recall_score(y_true, y_pred):.4f}')
        print(f'Precision: {precision_score(y_true, y_pred):.4f}')