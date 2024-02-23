from typing import List

import numpy as np
from models import Model

class SharedVocab:
    """
    Construct a shared vocabulary from a list of vocabs.
    Entries in this vocabulary are the union of all elements in the input vocabs.
    We then need to construct a two-way mapping:

    - input vocab -> shared vocab. This is an easy mapping to make.
    - shared vocab -> input vocab. There are two types here:
      - if a token is in both, the mapping is easy
      - if a token is in the shared vocab but not in the input vocab, we
        then tokenize it using the input vocab, to provide a unique tokenization.
    """

    def __init__(self, tokenizers: List):
        self.tokens_to_ids = {}
        self.ids_to_tokens = {}

        self.add_token("<pad>")
        self.add_token("<bos>")
        self.add_token("<eos>")

        # This maps from private vocab IDs to the shared vocab ID
        self.private_to_shared = []
        self.shared_tokenization = []
        self.tokenizers = tokenizers
        self.vocabs = []
        self.finalized = False
        for tokenizer in tokenizers:
            self.add_vocab(tokenizer.get_vocab())
        self.finalize()

    @property
    def pad_token_id(self):
        return self.tokens_to_ids["<pad>"]

    @property
    def bos_token_id(self):
        return self.tokens_to_ids["<bos>"]
    
    @property
    def eos_token_id(self):
        return self.tokens_to_ids["<eos>"]

    def __len__(self):
        return len(self.tokens_to_ids)

    def add_token(self, token: str):
        """
        Adds a token to the shared vocabulary and returns the token ID.
        """
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
        if self.finalized:
            raise ValueError("Cannot add vocab to a finalized shared vocab")

        # Add a copy of the vocab to the list of vocabs
        self.vocabs.append(vocab.copy())

        # Make sure every entry is in the shared vocab
        self.private_to_shared.append(np.full(len(vocab), -1, dtype=int))
        for i, token in enumerate(vocab):
            if token not in self.tokens_to_ids:
                # Create a new entry
                shared_token_id = self.add_token(token)
                self.private_to_shared[-1][i] = shared_token_id

    def finalize(self):
        """
        Finalize the shared vocab by returning projections from each vocab to the shared vocab.
        This facilitates direct interpolotion.
        """                
        self.finalized = True

        self.shared_tokenization = {}
        for token, token_id in self.tokens_to_ids.items():
            for vocab_i, vocab in enumerate(self.vocabs):
                if token not in vocab:
                    # Tokenize the token using the vocab
                    self.shared_tokenization[(vocab_i, token_id)] = self.tokenizers[vocab_i].encode(token, add_special_tokens=False)
                    # print(f"TOK({token}) in vocab {vocab_i} is {self.shared_tokenization[(vocab_i, token_id)]}")

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs into a string.
        """
        return " ".join(self.ids_to_tokens[token_id] for token_id in token_ids)

    def project_into(self, vocab_index, private_scores) -> List[float]:
        """
        Project scores from the input vocabulary specified by {vocab_index} to the shared vocab

        :param vocab_index: which input vocabulary to project from
        :param private_scores: scores in the input vocabulary
        """
        shared_scores = np.full(private_scores.shape, -np.inf, dtype=float)
        # for each row of private_scores, project the indices into the shared vocab
        print("SCORES SHAPE", shared_scores.shape)
        print("VOCAB", vocab_index, "SHAPE:", len(self.private_to_shared[vocab_index]))
        for beami in range(private_scores.shape[0]):
            for token_id, score in enumerate(private_scores[beami, :]):
                shared_token_id = self.private_to_shared[vocab_index][token_id]
                shared_scores[beami][shared_token_id] = score

        return shared_scores

    def project_outta(self, token_id, vocab_index) -> List[int]:
        """
        Project a token from the shared vocab to the input vocab specified by {vocab_index}
        """
        return self.shared_tokenization[token_id][vocab_index]


if __name__ == "__main__":
    import sys
    import argparse

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", "-m", type=str, nargs="+", default=["facebook/m2m100_418M", "facebook/nllb-200-distilled-600M"], help="Model names")
    parser.add_argument("--target-lang", "-t", type=str, default="fr", help="Target language")
    args = parser.parse_args()

    tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in args.model_names]

    vocab = SharedVocab(tokenizers)
    for name, tokenizer in zip(args.model_names, tokenizers):
        print(len(tokenizer.get_vocab()), name, sep="\t")
    print(len(vocab), "shared", sep="\t")

    # iterate over the first vocabulary until IDs fail
    token_id = 0
    while True:
        try:
            print(token_id, tokenizers[0].decode([token_id]))
            token_id += 1
        except KeyError:
            break
