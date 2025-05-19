class ModelCaller():
    def __init__(self, model_name):
        self.system_message = "You are a helpful assistant."
        self.model_name = model_name
    
    def pack_message(self, role, content):
        return {'role': role, 'content': content}