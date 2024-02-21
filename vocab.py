from typing import List
from models import Model

class SharedVocab:
    def __init__(self, vocabs: List):
        self.tokens_to_ids = {}
        self.ids_to_tokens = {}

        self.add_token("<pad>")
        self.add_token("<bos>")
        self.add_token("<eos>")

        self.vocabs = []
        for vocab in vocabs:
            self.add_vocab(vocab)

    @property
    def pad_token_id(self):
        return self.tokens_to_ids["<pad>"]

    @property
    def bos_token_id(self):
        return self.tokens_to_ids["<bos>"]
    
    @property
    def eos_token_id(self):
        return self.tokens_to_ids["<eos>"]

    def add_token(self, token):
        if token not in self.tokens_to_ids:
            # Create a new entry
            new_token_id = len(self.tokens_to_ids)
            self.tokens_to_ids[token] = new_token_id
            self.ids_to_tokens[new_token_id] = token

        return self.tokens_to_ids[token]

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
