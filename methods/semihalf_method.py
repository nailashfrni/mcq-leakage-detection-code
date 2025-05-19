from tqdm import tqdm
import numpy as np
import re
import time
import json

from methods.base_method import BaseMethod
from model_callers.model_caller import ModelCaller
from semihalf.model_utils import *

class SemihalfMethod(BaseMethod):
    def __init__(self, eval, prefix, model, tokenizer, mode, output_type, 
                compute_ppl, batch=0):
        super().__init__(eval, prefix, model, tokenizer, batch)
        self.mode = mode
        self.output_type = output_type
        self.compute_ppl = compute_ppl

    def evaluate(self, output_file):
        if self.output_type == 'csv':
            scores = []
            predictions = []
            row_softmaxes = []
            if self.compute_ppl:
                perplexities = []
        else:
            result = []
        
        start_time = time.time()
        counter = 0
        for idx, row in tqdm(self.eval.df.iterrows(), total=len(self.eval.df), desc=f"Processing Rows"):
            prompt, text_for_ppl, question = self.eval.format_prompt(row=row, mode=self.mode)
            if isinstance(self.model, ModelCaller):
                extracted_answer = None
                row_softmax = []
                trial = 0
                while extracted_answer not in ['A', 'B', 'C', 'D']:
                    response_text = self.model(prompt, temperature=0, max_tokens=2048)
                    regex = r"(?i)\s*([ABCD])\s*"
                    match = re.search(regex, response_text)
                    if match:
                        extracted_answer = match.group(1).strip()
                    trial += 1
                    if trial == 5:
                        extracted_answer = 'Failed'
                        break
            else:
                prompt, text_for_ppl, question = self.eval.format_prompt(row=row, mode=self.mode)
                extracted_answer, row_softmax = get_answer(self.model, self.tokenizer, prompt)
                # Extract the first character to match expected format (A, B, C, D)
                extracted_answer = extracted_answer[0].upper()

                generated_ppl = -20
                if self.compute_ppl:
                    generated_ppl = ppl(text_for_ppl, self.model, self.tokenizer)

            correct_answer = self.eval.get_row_answer(row).strip().upper()
            if self.output_type == 'csv':
                score = 1.0 if extracted_answer == correct_answer else 0.0
                scores.append(score)
                predictions.append(extracted_answer)
                row_softmaxes.append(row_softmax)
                if self.compute_ppl:
                    perplexities.append(generated_ppl)
            else:
                row_result = {
                    'id': self.eval.get_id(row),
                    'question': question,
                    'answer': self.eval.get_row_answer(row),
                    'extracted_answer': extracted_answer if extracted_answer else '',
                    'score': 1.0 if extracted_answer == self.eval.get_row_answer(row) else 0.0,
                    'row_softmax': row_softmax
                }
                label = self.eval.get_row_label(row)
                if label != -1:
                    row_result['gold_leakage'] = label
                if self.compute_ppl:
                    row_result['perplexity'] = generated_ppl
                result.append(row_result)
            counter += 1
            if counter in [10, 100, 1000, 10000]:
                print(f"--- SEMIHALF {counter} Instances: {(time.time() - start_time)} seconds ---")

        if self.output_type == 'csv':
            output_file += '.csv'
            self.eval.df['Extracted Answer'] = predictions
            self.eval.df['Score'] = scores
            self.eval.df['Row Softmax'] = row_softmaxes
            if self.compute_ppl:
                self.eval.df['Perplexity'] = perplexities
            print(f"Accuracy Score: {np.mean(scores)}")
            self.eval.df.to_csv(output_file, index=False)
            print('Save result to', output_file)
            if 'gold_leakage' in self.eval.df.columns:
                return self.eval.df['gold_leakage'], self.eval.df['Score']

        else:
            output_file += '.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print('Save result to', output_file)
            if 'gold_leakage' in result[0].keys():
                y_true = [d['gold_leakage'] for d in result]
                y_pred = [d['score'] for d in result]
                return y_true, y_pred

        return None, None