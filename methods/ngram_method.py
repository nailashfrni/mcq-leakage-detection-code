from rouge_score import rouge_scorer
from tqdm import tqdm
import time
import math
import torch
import json

from methods.base_method import BaseMethod
from ngram.metrics import *
from model_callers.model_caller import ModelCaller

class NgramMethod(BaseMethod):
    '''
    The code is adapted from https://github.com/GAIR-NLP/benbench
    '''
    def __init__(self, eval, prefix, model, tokenizer, base_model_dir='gpt-4o-mini', batch=0):
        super().__init__(eval, prefix, model, tokenizer, batch)
        self.base_model_dir = base_model_dir
        if self.tokenizer is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self, output_file):
        results = []

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        dataset = [row.to_dict() for _, row in self.eval.df.iterrows()]
        counter = 0
        start_time = time.time()
        for row in tqdm(dataset, total=len(dataset), desc=f"Processing Ngram"):
            result = self.calculate_n_gram_accuracy(row=row, scorer=scorer)
            results.append(result)
            counter += 1
            if counter in [10, 100, 1000, 10000]:
                print(f"--- N-GRAM {counter} Instances: {(time.time() - start_time)} seconds ---")

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print('Save result to', output_file)

        if 'gold_leakage' in results[0].keys():
            y_true = [d['gold_leakage'] for d in results]
            y_pred = [d['leakage'] for d in results]
            return y_true, y_pred
        return None, None

    def calculate_n_gram_accuracy(self, row, scorer=None):
        """
        Calculate n-gram accuracy for both OpenAI and Hugging Face models.
        """
        torch.cuda.empty_cache()
        row_options = self.eval.get_row_options(row)
        prefixes = [f"""{self.eval.get_row_question(row)}\nA. """,
                    "\nB. ",
                    "\nC. ",
                    "\nD. "]
        choices = ['A', 'B', 'C', 'D']
        full_text = f"{prefixes[0]}{row_options[0]}{prefixes[1]}{row_options[1]}{prefixes[2]}{row_options[2]}{prefixes[3]}{row_options[3]}"

        sample_results = {"idx": self.eval.get_id(row), 
                        "sample": full_text, 
                        "n_gram_results": []}
        
        label = self.eval.get_row_label(row)
        if label != -1:
            sample_results['gold_leakage'] = label

        sample_correct_n_grams = 0
        sample_total_n_grams = 0
        valid_exact_match, valid_edit_similarity, valid_rouge_score = 0, 0, 0

        for i in range(len(prefixes)):
            prefix = "".join(prefixes[:i+1]).strip()
            original_text = row_options[i]
            if (original_text == '') or (isinstance(original_text, float) and math.isnan(original_text)):
                original_text = ''
                predicted_text = ''
            elif isinstance(self.model, ModelCaller):
                len_ori_text = len(original_text.split())
                predicted_text = self.model(prefix, temperature=0, max_tokens=min(len_ori_text * 4, 20))
                predicted_text = ' '.join(predicted_text.replace(prefix.strip(), '')\
                                        .replace(f"{choices[i]}. ", "").split()[:len_ori_text]).strip()
            else:
                prefix_tokens = self.tokenizer.tokenize(prefix)
                try:
                    original_tokens = self.tokenizer.tokenize(original_text)
                except:
                    original_tokens = ["[EMPTY]"]
                encoding = self.tokenizer(
                    prefix,
                    is_split_into_words=False,
                    return_tensors="pt",
                    padding="longest"
                ).to(self.model.device)
                len_original_token = len(original_tokens)
                encoding['max_new_tokens'] = len_original_token + 1
                encoding['do_sample'] = False
                len_prefix = len(prefix_tokens)

                gens = self.model.generate(**encoding)

                predicted_ids = gens[0, len_prefix:].tolist()
                predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()

            # compute metric similarity
            edit_score = edit_similarity_score(predicted_text, original_text)
            rouge_score = rouge_l_score(predicted_text, original_text, scorer)
            exact_score = exact_match_score(predicted_text, original_text)

            valid_exact_match += exact_score
            valid_edit_similarity += int(edit_score > 0.75)
            valid_rouge_score += int(rouge_score > 0.75)

            n_gram_result = {
                "idx": int(i),
                "predicted_text": predicted_text,
                "original_text": original_text,
                "edit_similarity": edit_score,
                "rouge_score": rouge_score,
                "exact_match_score": exact_score
            }
            sample_results["n_gram_results"].append(n_gram_result)

            overall = {}
            overall["exact_match_correct_ratio"] = valid_exact_match / 4
            overall["edit_similarity_correct_ratio"] = valid_edit_similarity / 4
            overall["rouge_score_correct_ratio"] = valid_rouge_score / 4

            sample_results['overall'] = overall

            sample_total_n_grams += 1
            if original_text == predicted_text:
                sample_correct_n_grams += 1
            prefixes[i] += original_text

        # change the metric and threshold here
        sample_results['leakage'] = 1 if sample_results['overall']['rouge_score_correct_ratio'] >= 0.25 else 0
        return sample_results