import pandas as pd

class BaseEval():
    def __init__(self, input_file=None):
        self.formating_prompt = """Your response must ONLY IN the following format: '$LETTER' 
(without quotes) where LETTER is one of ABCD."""
        if input_file is not None:
            if '.csv' in input_file:
                self.df = pd.read_csv(input_file)
            else:
                self.df = pd.read_json(input_file)
        self.question_template = '{question}'
        self.option_template = """
A. {A}
B. {B}
C. {C}
D. {D}"""
        self.last_line_prompt = '\nAnswer: '
        self.question_col = 'question'
        self.id_col = 'id'

    def get_row_label(self, row):
        if 'gold_leakage' in row.keys():
            return row['gold_leakage']
        else:
            return -1

    def get_id(self,row):
        return row[self.id_col]

    def format_prompt(self, row, mode):
        main_prompt, row = self.format_main_prompt(row, mode, 
                                            self.question_template, 
                                            self.option_template)
        full_prompt = self.query_prompt + main_prompt + self.last_line_prompt
        return full_prompt, main_prompt, self.get_row_question(row)

    def format_main_prompt(self, row, mode, query_question, query_option):
        if mode == 'option_only':
            return query_option.format(**row)
        elif mode == 'semihalf':
            row[self.question_col] = self.get_truncated_sentence(self.get_row_question(row), 50)
        prompt = query_question.format(**row)
        if mode == 'question_only':
            return prompt
        prompt += query_option.format(**row)
        return prompt, row

    def get_truncated_sentence(self, sentence, percentage=100):
        '''
        sentece (str): sentence to be truncated
        percentage (float/int): how many percent the sentence would be returned
        '''
        words = sentence.split()
        num_to_retain = min(7, int(len(words) * (percentage / 100)))
        reduced_sentence = words[-num_to_retain:]
        return " ".join(reduced_sentence)

    def get_row_question(self, row):
        if 'Question' in row.keys():
            return row['Question']
        return row['question']
    
    def get_row_answer(self, row):
        if 'Answer' in row.keys():
            return row['Answer']
        return row['answer']

    def get_row_options(self, row):
        if 'A' in row.keys():
            return [row['A'], row['B'], row['C'], row['D']]
        return row['choices']