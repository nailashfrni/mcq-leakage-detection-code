import ast
from evals.base_eval import BaseEval

class HellaSwagEval(BaseEval):
    def __init__(self, input_file=None, is_close_weight=False):
        super().__init__(input_file)
        self.query_prompt = 'Given a context and 4 different endings, output the option letter (A, B, C, D) with the most likely ending.\n\n'
        if is_close_weight:
            self.query_prompt = f"""Given a context and 4 different endings, output the option letter 
(A, B, C, D) with the most likely ending. {self.formating_prompt}\n\n"""
        self.cpt_prompt = """HellaSwag Question
### Question: {}

### Answer: {}"""
        
        if input_file is not None:
            if '.csv' in input_file:
                self.df['endings'] = self.df['endings'].apply(lambda x: ast.literal_eval(x))
            if 'ctx' in self.df.columns:
                self.question_template = '{ctx}'
                self.question_col = 'ctx'


    def format_prompt(self, row, mode):
        """Format prompt given a row from a dataset."""
        if 'A' not in row.keys():
            choices = ['A', 'B', 'C', 'D']
            options = self.get_row_options(row)
            for i in range(len(choices)):
                row[choices[i]] = options[i]
        return super().format_prompt(row, mode)

    def get_row_question(self, row):
        if 'ctx' in row.keys():
            return row['ctx']
        return super().get_row_question(row)

    def get_row_answer(self, row):
        chars = ['A', 'B', 'C', 'D']
        if 'label' in row.keys():
            return chars[row['label']]
        return super().get_row_answer(row)

    def get_row_options(self, row):
        if 'endings' in row.keys():
            return row['endings']
        return super().get_row_options(row)

    def __str__(self):
        return 'hellaswag'