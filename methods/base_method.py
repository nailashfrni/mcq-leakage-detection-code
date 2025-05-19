import pandas as pd

from utils import load_model

class BaseMethod:
    def __init__(self, eval, prefix, model, tokenizer, batch=0):
        self.eval = eval
        self.batch = batch
        self.prefix = prefix
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, output_file):
        pass