from evals.base_eval import BaseEval

class MMLUEval(BaseEval):
    def __init__(self, input_file=None, is_close_weight=False):
        super().__init__(input_file)
        self.query_prompt = 'Answer the following multiple choice question.\n\nQuestion: '
        if is_close_weight:
            self.query_prompt = f"""Answer the following multiple choice question. {self.formating_prompt}\n\nQuestion:"""
        self.cpt_prompt = """MMLU Question
### Question: {}

### Answer: {}"""

        # QUERY_PROMPT_IFT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        # ### Instruction:
        # Answer the following multiple choice question.

        # ### Input:"""
        # LAST_LINE_PROMPT_IFT = """\n\n### Response: """

    def format_prompt(self, row, mode):
        """Format prompt given a row from MMLU dataset."""
        if 'A' not in row.keys():
            choices = ['A', 'B', 'C', 'D']
            for i in range(len(choices)):
                row[choices[i]] = row['choices'][i]
        if 'Question' in row.keys():
            row['question'] = row['Question']

        return super().format_prompt(row, mode)

    def __str__(self):
        return 'mmlu'