class SharedVocab:
    def __init__(self, models):
        self.models = models
        self.pad_token_id = 0
        self.eos_token_id = 1

        self.ids_to_tokens = {
            0: "<pad>",
            1: "<eos>",
        }
        self.tokens_to_ids = {
            "<pad>": 0,
            "<eos>": 1,
        }
        self.vocabs = []

    def add_vocab(self, vocab):
        """
        Add the vocabulary to the shared vocab.
        """
        # Add a copy of the vocab to the list of vocabs
        self.vocabs.append(vocab.copy())

        for token in vocab:
            if token not in self.tokens_to_ids:
                # Create a new entry
                new_token_id = len(self.tokens_to_ids)
                self.tokens_to_ids[token] = new_token_id
                self.ids_to_tokens[new_token_id] = token

        for token in self.tokens_to_ids.keys():
            if token not in vocab:
                pass

    def finalize(self):
        """
        Finalize the shared vocab by returning projections from each vocab to the shared vocab.
        This facilitates direct interpolotion.
        """
