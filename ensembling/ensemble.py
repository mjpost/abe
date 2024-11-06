import os, sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")


from typing import Optional, Tuple, Union, List, Dict, Any


from models import get_models, Model
from utils import Trie
from search import beam_search, get_pad_beams

import torch
from transformers.generation.utils import (
    GenerateBeamOutput, 
)

import math
import json
from collections import defaultdict

def ensemble_beam_search(
            batch: List[dict],
            models: List[Model],
            weights: List[float],
            num_beams: int = 5,
            max_length : int = -1,
            trie : Trie = None) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    Adapted from `~transformers.generation_utils.GenerationMixin.beam_search` to accept a list of input_ids

    - Code source:
        https://github.com/huggingface/transformers/blob/07e3454f034b4889925621e8e3253547d2a04aa7/src/transformers/generation/utils.py#L2764
    - Beam search support:
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py
    """

    num_models = len(models)
    device = models[0].model.device
    batch_size = len(batch[0])
    batch_beam_size = batch_size * num_beams

    for model_i, model in enumerate(models):
        # Initialize each model with the input
        model.set_input(batch[model_i], num_beams, max_length)
    
    # Hypotheses on beams across models are always consistent, but one may be shorter than another.
    # We keep track of which model is stalled for any given beam/hypothesis item
    stalled_states = [[False for _ in range(num_models)] for _ in range(batch_beam_size)]

    beam_completed = [[] for _ in range(batch_size)]  # contains completed sentences for each batch item

    continue_search = [True for _ in range(batch_size)]
    step_i = 0

    while any(continue_search):
        paired_outputs = defaultdict(lambda: defaultdict(list)) # each beam item has a list of outputs for each model
        cached_steps = []

        # Each model still steps individually
        for model_i, model in enumerate(models):

            # Right now we step on all items -- even if model is stalled.
            # TODO: In the future, there may be room for improvement in efficiency here
            step_outputs, next_token_scores = model.step()

            # Cache the transformer decoder statement for faster decoding
            cached_steps.append(step_outputs)
            
            # Some debugging prints
            logger.debug(f"STEP {step_i} ({type(model.model)}) MODEL {model_i} STALL {stalled_states}")
            for line in model.get_beam_string(step_i, model_i):
                logger.debug(line)

            for beam_i, beam_expansion in enumerate(next_token_scores):
                if model.target_tokenizer.pad_token_id:
                    pad_token_id = model.target_tokenizer.pad_token_id
                else:
                    pad_token_id = model.target_tokenizer.eos_token_id
                # paired_outputs[beam_i][model_i] = get_sorted_output_extensions(
                #     stalled_states[beam_i][model_i],
                #     beam_expansion,
                #     model.beam_scores[beam_i],
                #     model.target_tokenizer.pad_token_id,
                #     device=device,
                #     trie=trie
                # )
                paired_outputs[beam_i][model_i] = get_sorted_output_extensions(
                    stalled_states[beam_i][model_i],
                    beam_expansion,
                    model.beam_scores[beam_i],
                    pad_token_id,
                    device=device,
                    trie=trie
                )

        # All models have stepped. Start search by seeding the heap with the best candidates from each beam
        next_beam = []
        for batch_i in range(batch_size):
            if continue_search[batch_i]:

                # This is our current best hypothesis. We don't want to continue our search if we every pass this score
                max_score = max(beam_completed[batch_i]).raw_score() if len(beam_completed[batch_i]) > 0 else -math.inf
                next_batch_beam, completed_beams = beam_search(
                    batch_offset = batch_i * num_beams,
                    num_beams = num_beams,
                    paired_outputs = paired_outputs,
                    models = models,
                    weights = weights,
                    completed_beams = len(beam_completed[batch_i]),
                    stalled_states = stalled_states[batch_i * num_beams: (batch_i + 1) * num_beams],
                    max_length = max_length,
                    max_score = max_score
                )

                # if the search returned less then the number of beams, there was some early stop criteria
                # this is either due to a max_length or max_score being reached
                if len(next_batch_beam) < num_beams:

                    # if there are no beams to continue, we have stopped without any valid continuations,
                    # so we will stop searching on this batch item---whatever is in the completed itmes is the end results
                    if len(next_batch_beam) == 0:
                        continue_search[batch_i] = False

                    # Otherwise we need to pad our beam with some empty items to the model is not confused by the smaller beam size
                    next_batch_beam = get_pad_beams(next_batch_beam, models, batch_i, num_beams, weights)
                    next_beam.extend(next_batch_beam)
                    beam_completed[batch_i].extend(completed_beams)
                    continue

                # If we have any items that have been completed, we need to add them to the completed list
                beam_completed[batch_i].extend(completed_beams)
                logger.debug(f"BEAM {batch_i} COMPLETED {len(beam_completed[batch_i])} BEAMS")

                # If we've completed enough beams then we quit searching
                if len(beam_completed[batch_i]) == num_beams:
                    continue_search[batch_i] = False

                # or if the best hypothesis is already worse than the worst completed beam
                if len(beam_completed[batch_i]) > 0 and next_batch_beam[0][0].raw_score() < max(beam_completed[batch_i]).raw_score():
                    continue_search[batch_i] = False

                for beam_i, beam in enumerate(next_batch_beam):
                    candidates = [beam[0].outputs[_][1].token for _ in range(num_models)]
                    candidates_strings = " ||| ".join(candidates)
                    logger.debug(f"SELECTED {beam_i} {candidates_strings}")
            else:
                # if this batch item is done, we need to pad the beam with empty items
                # this is due to having the batch_beam_size
                next_batch_beam = get_pad_beams([], models, batch_i, num_beams, weights)

            next_beam.extend(next_batch_beam)

        # after all our batch items have been extended, we'll update the models
        stalled_states = update_models_with_beams(
            next_beam,
            models,
            cached_steps                
            )
        step_i += 1


    outputs = []
    for batch_i, completed in enumerate(beam_completed):
        sorted_completions = sorted(completed, key=lambda x: x.raw_score(), reverse=True)
        best_completion = sorted_completions[0]
        scores = [_.item() for _ in best_completion.scores]
        output_tokens = [model.target_tokenizer.convert_ids_to_tokens(best_completion.output_ids[model_i]) for model_i, model in enumerate(models)]
        combined_score = best_completion.raw_score().item()
        out_str = models[0].target_tokenizer.decode(best_completion.output_ids[0], skip_special_tokens=True)
       
        for model_i, model in enumerate(models):
            ids = best_completion.output_ids[model_i]
            tokens = model.target_tokenizer.convert_ids_to_tokens(ids)
            logger.debug(f"MODEL {model_i}")
            logger.debug(f"IDS: {ids}")
            logger.debug(f"TOKS: {tokens}")

        for completion_j, completion in enumerate(sorted_completions):
            j_out_str = models[0].target_tokenizer.decode(completion.output_ids[0], skip_special_tokens=True)
            j_scores = [_.item() for _ in completion.scores]
            j_combined_score = completion.raw_score().item()
            logger.debug(f"COMPLETION {batch_i} {completion_j} {j_out_str} {j_scores} {j_combined_score}")

        input_ids = [model.input_ids[batch_i].tolist() for model in models]
        
        outputs.append({
            "input_ids": input_ids,
            "sequence": out_str,
            "scores": scores,
            "combined_score": combined_score,
            "token_scores": best_completion.token_scores,
            "tokens": output_tokens,
            "token_ids": best_completion.output_ids,
            "weights": weights
        })

    return outputs



def update_models_with_beams(
        next_beam,
        models,
        cached_steps
):
    
    beam_size = len(next_beam)
    stalled_states = [[] for _ in range(beam_size)]
    for model_i, model in enumerate(models):
        beam_indices = [next_beam[beam_j][0].beam_index for beam_j in range(beam_size)]
        beam_tokens = [next_beam[beam_j][0].outputs[model_i][1].idx.item() for beam_j in range(beam_size)]
        beam_scores = [next_beam[beam_j][0].outputs[model_i][1].score for beam_j in range(beam_size)]
        update_mask = [next_beam[beam_j][1][model_i] for beam_j in range(beam_size)]
        for beam_j in range(beam_size):
            stalled_states[beam_j].append(update_mask[beam_j])
        model.update_beam(beam_indices, beam_tokens, beam_scores, step_outputs=cached_steps[model_i])
    
    return stalled_states

def get_sorted_output_extensions(
        stalled,
        next_token_scores,
        beam_score,
        pad_token_id,
        device : torch.device = torch.device('cpu'),
        trie: Optional[Trie] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if stalled:
        return [[beam_score], torch.tensor([pad_token_id], dtype=torch.long, device=device)]
    
    # If a trie was constructed, select the top-k tokens that are in the trie (limits sort and search space)
    if trie:
        pass

    # need to somehow expand this
    return torch.sort(next_token_scores + beam_score, descending=True)



def batch_generator(istream, batch_size, num_models):
    batch = [[] for _ in range(num_models)]
    for line_i, line in enumerate(istream):
        line = line.split('\t')
        assert len(line) == num_models, f"Line {line_i} does not have {num_models} columns"
        for model_i, l in enumerate(line):
            batch[model_i].append(json.loads(l.strip()))
        if len(batch[0]) == batch_size:
            yield batch
            batch = [[] for _ in range(num_models)]
    if len(batch[0]) > 0:
        yield batch

def build_output(output, args):
    if args.score:
        return json.dumps(output, ensure_ascii=False)
    else:
        return output["sequence"]

def print_output(outputs, args, ostream):
    for o in outputs:
        print(build_output(o, args), file=ostream)


def ensemble_models(args):
    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    logging.debug(f"Using device: {device}")

    models = get_models(args.models, device, args.cache)
    weights = [w / sum(args.weights) for w in args.weights] if args.weights is not None else [1/len(models) for _ in models]
    trie = Trie() if args.trie else None

    istream = open(args.input, 'r') if args.input else sys.stdin

    ostream = open(args.output, 'w') if args.output else sys.stdout

    batches = batch_generator(istream, args.batch_size, len(models))
   

    

    
    outputs_formatted = [[] for i in range(7)]
    for i, batch in enumerate(batches):
        outputs = ensemble_beam_search(
                    batch,
                    models,
                    weights,
                    num_beams=args.num_beams,
                    max_length=args.max_length,
                    trie=trie)
        print_output(outputs, args, ostream)
        
        outputs_formatted[i] = outputs
        
    

    
    with open(args.output+'.jsonl', 'w', encoding='utf8') as file:
        for line in outputs_formatted:
            json.dump(line, file, ensure_ascii=False)
            file.write('\n')
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble models')
    
    parser.add_argument("--input", '-i', type=str, help='Input file. Defaults to stdin', default='input-test')
    parser.add_argument("--output", '-o', type=str, help='Output file. Defaults to stdout', default='output-test')

    parser.add_argument("--models", '-m', type=str, help='Models to ensemble', nargs='+', default=["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"])
    parser.add_argument("--weights", '-w', type=float, help='Weights for each model', nargs='+')

    parser.add_argument("--num-beams", '-b', type=int, help='Number of beams for beam search', default=5)
    parser.add_argument("--batch-size", '-t', type=int, help='Batch size for inference', default=1)
    parser.add_argument("--max-length", '-l', type=int, help='Maximum length of the output', default=100)
    parser.add_argument("--score", '-s', action='store_true', help='Output the score of each model')

    parser.add_argument("--trie", default=False, action='store_true', help='Use trie for finding extensions')

    parser.add_argument("--cpu", '-c', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument("--cache", '-a', action='store_true', help='Cache the models')

    parser.add_argument("--debug", '-d', action='store_true', help='Debug mode')

    args = parser.parse_args()


    if args.debug:
        logging.getLogger("ensembling").setLevel(logging.DEBUG)

    if args.weights is not None:
        assert len(args.models) == len(args.weights), "Number of models and weights must be the same"
        assert all([w >= 0 for w in args.weights]), "Weights must be non-negative"
    assert args.num_beams > 0, "Number of beams must be positive"
    assert args.batch_size > 0, "Batch size must be positive"

    ensemble_models(args)



    
    

    
    
    
    
