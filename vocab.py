from collections import Counter
from typing import List

import sys
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

    BOS = "<s>"
    PAD = "<pad>"
    EOS = "</s>"
    UNK = "<unk>"

    def __init__(self, tokenizers: List):
        self.tokens_to_ids = {}
        self.ids_to_tokens = {}

        self.add_token(self.BOS)
        self.add_token(self.PAD)
        self.add_token(self.EOS)
        self.add_token(self.UNK)

        # This maps from private vocab IDs to the shared vocab ID
        self.private_to_shared = []
        self.shared_tokenization = {}
        self.tokenizers = tokenizers
        self.vocabs = []
        self.finalized = False
        for tokenizer in tokenizers:
            self.add_vocab(tokenizer.get_vocab())
        self.finalize()

    @property
    def pad_token_id(self):
        return self.tokens_to_ids[self.PAD]

    @property
    def bos_token_id(self):
        return self.tokens_to_ids[self.BOS]
    
    @property
    def eos_token_id(self):
        return self.tokens_to_ids[self.EOS]

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

    def add_vocab(self, private_vocab):
        """
        Add the vocabulary to the shared vocab.
        """
        if self.finalized:
            raise ValueError("Cannot add vocab to a finalized shared vocab")

        # Add a copy of the vocab to the list of vocabs
        self.vocabs.append(private_vocab.copy())

        # Make sure every entry is in the shared vocab
        self.private_to_shared.append(np.full(len(private_vocab), -1, dtype=int))
        new_token_count = 0
        for i, token in enumerate(private_vocab):
            if token not in self.tokens_to_ids:
                # Create a new entry
                self.add_token(token)
                new_token_count += 1
            shared_token_id = self.tokens_to_ids[token]
            self.private_to_shared[-1][i] = shared_token_id
            # print("VOCAB", len(self.vocabs) - 1, i, token, shared_token_id, self.decode([shared_token_id]))
        print(f"{new_token_count}/{len(private_vocab)}={new_token_count/len(private_vocab):.2%} new tokens in vocab {len(self.vocabs) - 1}", file=sys.stderr)


    def finalize(self):
        """
        Finalize the shared vocab by returning projections from each vocab to the shared vocab.
        This facilitates direct interpolotion.
        """                
        self.finalized = True

        self.shared_tokenization = {}
        for token, token_id in self.tokens_to_ids.items():
            for vocab_i, vocab in enumerate(self.vocabs):
                if token in vocab:
                    self.shared_tokenization[(vocab_i, token_id)] = [vocab[token]]
                else:
                    # Tokenize the token using the vocab
                    self.shared_tokenization[(vocab_i, token_id)] = self.tokenizers[vocab_i].encode(token, add_special_tokens=False)
                    # print(f"TOK({token}) in vocab {vocab_i} is {self.shared_tokenization[(vocab_i, token_id)]}")

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs into a string.
        """
        return " ".join(self.ids_to_tokens[token_id] for token_id in token_ids)

    def project_into(self, 
                     vocab_index, 
                     private_scores, 
                     history=None,
                     step=None, 
                     default_value=0.0) -> List[float]:
        """
        Project scores from the input vocabulary specified by {vocab_index} to the shared vocab

        :param vocab_index: which input vocabulary to project from
        :param private_scores: scores in the input vocabulary
        :param history: Model's complete token history for the entire beam
        :param step: The beam step number, used for logging
        :param default_value: The default value to use for tokens that are not in the shared vocab
        """
        num_beams = private_scores.shape[0]
        shared_vocab_size = len(self)
        shared_scores = np.full((num_beams, shared_vocab_size), default_value, dtype=float)

        # for each row of private_scores, project the indices into the shared vocab
        # print("SCORES SHAPE", shared_scores.shape)
        # print("VOCAB", vocab_index, "SHAPE:", len(self.private_to_shared[vocab_index]))
        logfile = open(f"log.project_into_{vocab_index}_step{step}", "w")
        for beami in range(num_beams):
            for token_id, score in enumerate(private_scores[beami, :]):
                if token_id >= len(self.private_to_shared[vocab_index]):
                    # print(f"Skipping token {vocab_index}/{token_id} (= {score}) which is out of bounds")
                    break
                shared_token_id = self.private_to_shared[vocab_index][token_id]
                if shared_token_id != -1:
                    shared_scores[beami][shared_token_id] = score

                # write as a floating point number (not scientific notation)
                if beami == 0 and shared_token_id != -1:
                    print(f"{float(score):.10f}", shared_token_id, self.decode([shared_token_id]), file=logfile)

        return shared_scores

    def project_outta(self, token_id, vocab_index) -> List[int]:
        """
        Project a token from the shared vocab to the input vocab specified by {vocab_index}
        """
        return self.shared_tokenization[(int(vocab_index), int(token_id))]


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
