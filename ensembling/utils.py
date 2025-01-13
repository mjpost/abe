import os, sys
import logging
import torch
import math

from typing import List

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")


class Node:
    def __init__(self, idx=None, byte_string=None):
        self.idx = [] if idx is None else idx
        self.children = [None] * 256
        self.byte_string = [] if byte_string is None else byte_string

    def add_child(self, child, byte_id):
        self.children[byte_id] = child

    def enumerate_children(self):
        for child in self.children:
            if child is not None:
                for subchild in child.enumerate_children():
                    yield subchild
                yield child


class Trie:
    def __init__(self, vocab_length):
        self.root = Node()
        self.vocab_length = vocab_length

    def add_string(self, token_string, token_id):
        node = self.root
        byte_sequence = token_string

        # traverse the tree, adding nodes where necessary
        for byte_id in byte_sequence:
            if node.children[byte_id] is None:
                node.children[byte_id] = Node(byte_string=node.byte_string + [byte_id])
            node = node.children[byte_id]

        # set the last node to the token_id
        node.idx.append(token_id)

    def search_key(self, affix_string, allow_root=True):
        if len(affix_string) == 0:
            return torch.ones(self.vocab_length, dtype=torch.bool)

        mask = torch.zeros(self.vocab_length, dtype=torch.bool)
        byte_sequence = affix_string

        node = self.root
        if allow_root:
            for id in node.idx:
                mask[id] = 1

        add_children = True
        for byte_id in byte_sequence:
            if node.children[byte_id] is None:
                add_children = False
                break

            node = node.children[byte_id]
            for id in node.idx:
                mask[id] = 1


        if add_children:
            for child in node.enumerate_children():
                for id in child.idx:
                    mask[id] = 1

        return mask
    
    def search_key_inf_mask(self, affix_string, allow_root=True):
        if len(affix_string) == 0:
            return torch.zeros(self.vocab_length, dtype=torch.bool)

        mask = torch.ones(self.vocab_length, dtype=torch.bool) * -math.inf
        byte_sequence = affix_string

        node = self.root
        if allow_root:
            for id in node.idx:
                mask[id] = 0

        add_children = True
        for byte_id in byte_sequence:
            if node.children[byte_id] is None:
                add_children = False
                break

            node = node.children[byte_id]
            for id in node.idx:
                mask[id] = 0

        if add_children:
            for child in node.enumerate_children():
                for id in child.idx:
                    mask[id] = 0

        return mask
    
    def search_key_indices(self, affix_string, allow_root = True):
        indices = []
        byte_sequence = affix_string

        node = self.root
        if allow_root or len(byte_sequence) == 0:
            for id in node.idx:
                indices += node.idx

        add_children = True
        for byte_id in byte_sequence:
            if node.children[byte_id] is None:
                add_children = False
                break

            node = node.children[byte_id]
            indices += node.idx

        if add_children:
            for child in node.enumerate_children():
                indices += child.idx

        return list(set(indices))


def compatibility(models, next_state):
    # 0 means compatible and all finished
    # 1 means compatible
    # -1 means incompatible
    num_models = len(models)
    candidate_strings = []
    for model_i, model in enumerate(models):
        candidate_strings.append(model.extend_beam_string(next_state.beam_index, next_state.outputs[model_i][1].idx))

    # first we check if there are sentences that have ended
    eos_endings = [model.is_eos(next_state.outputs[model_i][1].idx) for model_i, model in enumerate(models)]

    # if any strings have ended, compatibility must be:
    # - exactly equal for finished strings
    # - substring only for unfinished strings
    if any(eos_endings):
        finished_strings = [candidate_strings[i] for i in range(num_models) if eos_endings[i]]
        unfinished_strings = [candidate_strings[i] for i in range(num_models) if not eos_endings[i]]
        leading_string = finished_strings[0]
        compatbilities = [leading_string == fin for fin in finished_strings] \
            + [leading_string.startswith(ufin) for ufin in unfinished_strings]
        if all(compatbilities):
            if all(eos_endings):
                return 0, [True for _ in models], [None for _ in range(num_models)] # compatible, no stalling, all models complete
            else:
                postfixes = [leading_string[len(_):] for _ in candidate_strings]
                return 1, eos_endings, postfixes# compatibile, finished strings are stalled
        else:
            return -1, None, None# incompatible

    # String lengths will determine which string is the "leading" string
    string_lengths = [len(candidate_strings[i]) for i in range(num_models)]
    max_length = max(string_lengths)
    min_length = min(string_lengths)

    leading_string = candidate_strings[string_lengths.index(max_length)]
    compatibilities = [leading_string.startswith(candidate_strings[i]) for i in range(num_models)]
    if all(compatibilities):
        postfixes = [leading_string[len(_):] for _ in candidate_strings]
        if max_length == min_length:
            ret_val = [False for _ in range(num_models)]
            return 1, ret_val, postfixes
        else:
            ret_val = [l == max_length for l in string_lengths]
            return 1, ret_val, postfixes
    else:
        return -1, None, None


def tokenize(tokenizer, bos_tokens=None, inputs=None, eos=True):
    out = []
    if bos_tokens is not None:
        out += tokenizer.convert_tokens_to_ids(bos_tokens)
        if tokenizer.bos_token_id not in out and tokenizer.unk_token_id in out:
            logger.error(f"UNK token found in BOS tokens")
            sys.exit(-1)
        logger.debug(f"ids for BOS: {out}")
    if inputs is not None:
        if eos:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs) + [tokenizer.eos_token])
        else:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs))
        out += tokens
        logger.debug(f"Tokenized input: {tokens}")
    return out


SPIECE_UNDERLINE = "▁"
GPIECE = "Ġ"
BYTE_MAP = ["<0x" + f"{i:02x}>".upper() for i in range(256)]
TOKENIZER_CONFIG = {
    "facebook/nllb-200-distilled-600M": {
        "lstrip": False,
        "special_character": SPIECE_UNDERLINE,
        "begin_word": True,
        "byte_map": BYTE_MAP,
        "add_space": True
    },
    "facebook/m2m100_418M": {
        "lstrip": False,
        "special_character": SPIECE_UNDERLINE,
        "begin_word": True,
        "byte_map": BYTE_MAP,
        "add_space": True
    },
    "rewicks/baseline_en-de_64k_ep25": {
        "lstrip": False,
        "special_character": SPIECE_UNDERLINE,
        "begin_word": True,
        "byte_map": BYTE_MAP,
        "add_space": True
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "lstrip": False,
        "special_character": GPIECE,
        "begin_word": True,
        "byte_map": BYTE_MAP,
        "add_space": False
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "lstrip": False,
        "special_character": GPIECE,
        "begin_word": True,
        "byte_map": BYTE_MAP,
        "add_space": False
    }
}