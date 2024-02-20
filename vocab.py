class SharedVocab:
    def __init__(self, models):
        self.models = models
        self.pad_token_id = 0
        self.eos_token_id = 0

