import os, sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")

class Trie:
    pass

def compatibility(models, next_state):
    # 0 means compatible and all finished
    # 1 means compatible
    # -1 means incompatible
    num_models = len(models)
    candidate_strings = []
    for model_i, model in enumerate(models):
        if next_state.outputs[model_i][1].idx.item() == 22:
            pass

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
                return 0, [True for _ in models] # compatible, no stalling, all models complete
            else:
                return 1, eos_endings # compatibile, finished strings are stalled
        else:
            return -1, None # incompatible

    # String lengths will determine which string is the "leading" string
    string_lengths = [len(candidate_strings[i]) for i in range(num_models)]
    max_length = max(string_lengths)
    min_length = min(string_lengths)

    leading_string = candidate_strings[string_lengths.index(max_length)]
    compatibilities = [leading_string.startswith(candidate_strings[i]) for i in range(num_models)]
    if all(compatibilities):
        if max_length == min_length:
            ret_val = [False for _ in range(num_models)]
            return 1, ret_val
        else:
            ret_val = [l == max_length for l in string_lengths]
            return 1, ret_val
    else:
        return -1, None


def tokenize(tokenizer, bos_tokens=None, inputs=None):
    out = []
    if bos_tokens is not None:
        out += tokenizer.convert_tokens_to_ids(bos_tokens)
        if tokenizer.unk_token_id in out:
            logger.error(f"UNK token found in BOS tokens")
            sys.exit(-1)
        logger.debug(f"ids for BOS: {out}")
    if inputs is not None:
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs) + [tokenizer.eos_token])
        out += tokens
        logger.debug(f"Tokenized input: {tokens}")
    return out