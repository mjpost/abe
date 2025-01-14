
########################################################################################################
#                                                                                                      #
#                                         PACKAGING AND LOGGING                                        #
#                                                                                                      #
########################################################################################################

import os, sys
import logging
import pathlib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")

if (__package__ is None or __package__ == "") and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'ensembling'

########################################################################################################
#                                                                                                      #
#                                               IMPORTS                                                #
#                                                                                                      #
########################################################################################################

from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import math
import json
from collections import defaultdict

from ensembling.models import get_models, Model, build_tries
from ensembling.search import beam_search, get_pad_beams, sample_search


########################################################################################################
#                                                                                                      #
#                                            BEAM SEARCH                                               #
#                                                                                                      #
########################################################################################################


def ensemble_beam_search(
            batch: List[dict],
            models: List[Model],
            weights: List[float],
            num_beams: int = 5,
            max_length : int = -1,
            trie: bool = False):
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
    postfixes = [["".encode('utf-8') for _ in range(num_models)] for _ in range(batch_beam_size)]

    beam_completed = [[] for _ in range(batch_size)]  # contains completed sentences for each batch item

    continue_search = [True for _ in range(batch_size)]
    step_i = 0

    while any(continue_search):
        paired_outputs = defaultdict(lambda: defaultdict(list)) # each beam item has a list of outputs for each model
        cached_steps = []

        # This stops if our models are at their maximum generation length
        force_stop = False
        for model in models:
            if model.output_ids.shape[1] == max_length:
                force_stop = True

        # Each model still steps individually
        for model_i, model in enumerate(models):

            # Right now we step on all items -- even if model is stalled.
            step_outputs, next_token_scores = model.step()

            # Cache the transformer decoder statement for faster decoding
            cached_steps.append(step_outputs)
            
            # Some debugging prints
            logger.debug(f"STEP {step_i} ({model.model_name}) MODEL {model_i} STALL {stalled_states}")
            for line in model.get_logging_string(step_i, model_i):
                logger.debug(line)

            for beam_i, beam_expansion in enumerate(next_token_scores):
                paired_outputs[beam_i][model_i] = get_sorted_output_extensions(
                    stalled_states[beam_i][model_i],
                    beam_expansion,
                    model.beam_scores[beam_i],
                    model.eos_token_ids[0],
                    device=device,
                    model=model,
                    trie=trie,
                    postfix=postfixes[beam_i][model_i],
                    force_stop = force_stop
                )

        # All models have stepped. Start search by seeding the heap with the best candidates from each beam
        next_beam = []
        for batch_i in range(batch_size):
            if continue_search[batch_i]:

                # This is our current best hypothesis. We don't want to continue our search if we every pass this score
                max_score = max(beam_completed[batch_i]).score() if len(beam_completed[batch_i]) > 0 else -math.inf
                next_batch_beam, completed_beams = beam_search(
                    batch_offset = batch_i * num_beams,
                    num_beams = num_beams,
                    paired_outputs = paired_outputs,
                    models = models,
                    weights = weights,
                    completed_beams = len(beam_completed[batch_i]),
                    stalled_states = stalled_states[batch_i * num_beams: (batch_i + 1) * num_beams],
                    max_length = max_length,
                    max_score = max_score,
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
                if len(beam_completed[batch_i]) > 0 and next_batch_beam[0][0].score() < max(beam_completed[batch_i]).score():
                    continue_search[batch_i] = False

                for beam_i, beam in enumerate(next_batch_beam):
                    candidates = [beam[0].outputs[_][1].token for _ in range(num_models)]
                    candidates_strings = " ||| ".join(candidates)
                    postfix_strings = b"|||".join(beam[2])
                    logger.debug(f"SELECTED {beam_i} {candidates_strings}")
                    logger.debug(f"POSTFIXES: {postfix_strings}")
            else:
                # if this batch item is done, we need to pad the beam with empty items
                # this is due to having the batch_beam_size
                next_batch_beam = get_pad_beams([], models, batch_i, num_beams, weights)

            next_beam.extend(next_batch_beam)

        # after all our batch items have been extended, we'll update the models
        stalled_states, postfixes = update_models_with_beams(
            next_beam,
            models,
            cached_steps,
            last_stalled_states=stalled_states              
            )
        step_i += 1


    outputs = []
    for batch_i, completed in enumerate(beam_completed):
        # TODO: if a hypothesis is incomplete (because max length has been reached, this will error due to the padding values)
        # TODO: fill in the dummy function in utils and call here when applicable

        sorted_completions = sorted(completed, key=lambda x: x.score(), reverse=True)
        if len(sorted_completions) == 0:
            logger.info("Unable to find any completions under given criteria. Attempt to increase your maximum size and try again")
            outputs.append({})
            continue
        best_completion = sorted_completions[0]
        scores = [_.item() for _ in best_completion.scores]
        output_tokens = [model.target_tokenizer.convert_ids_to_tokens(best_completion.output_ids[model_i]) for model_i, model in enumerate(models)]
        combined_score = best_completion.raw_score().item()
        out_str = models[0].target_tokenizer.decode(best_completion.output_ids[0][-len(best_completion.token_scores[0]):], skip_special_tokens=True).strip()
       
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

        input_ids = [model.input_ids[batch_i].tolist() for model in models if model.is_encoder_decoder]
        
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
        cached_steps,
        last_stalled_states
):
    
    beam_size = len(next_beam)
    stalled_states = [[] for _ in range(beam_size)]
    next_postfixes = [[] for _ in range(beam_size)]
    for model_i, model in enumerate(models):
        beam_indices = [next_beam[beam_j][0].beam_index for beam_j in range(beam_size)]
        beam_tokens = [next_beam[beam_j][0].outputs[model_i][1].idx.item() for beam_j in range(beam_size)]
        beam_scores = [next_beam[beam_j][0].outputs[model_i][1].score for beam_j in range(beam_size)]
        update_mask = [next_beam[beam_j][1][model_i] for beam_j in range(beam_size)]
        postfix = [next_beam[beam_j][2][model_i] for beam_j in range(beam_size)]
        for beam_j in range(beam_size):
            stalled_states[beam_j].append(update_mask[beam_j])
            next_postfixes[beam_j].append(postfix[beam_j])
        model.update_beam(beam_indices, beam_tokens, beam_scores, step_outputs=cached_steps[model_i])
    
    return stalled_states, next_postfixes

def get_sorted_output_extensions(
        stalled,
        next_token_scores,
        beam_score,
        eos_token_id,
        device : torch.device = torch.device('cpu'),
        trie: bool = False,
        model: Model = None,
        postfix : str = None,
        force_stop : bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if force_stop:
        return [[(next_token_scores + beam_score)[eos_token_id]], torch.tensor([eos_token_id], dtype=torch.long, device=device)]

    if stalled:
        return [[beam_score], torch.tensor([-1], dtype=torch.long, device=device)]
    
    # If a trie was constructed, select the top-k tokens that are in the trie (limits sort and search space)
    if trie:
        if postfix != "":
            mask = model.trie.search_key_inf_mask(postfix).to(device)
            return torch.sort((next_token_scores + mask + beam_score), descending=True)

    # need to somehow expand this
    return torch.sort(next_token_scores + beam_score, descending=True)



def ensemble_sample(
            batch: List[dict],
            models: List[Model],
            weights: List[float],
            num_samples: int = 1,
            max_length : int = -1,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.95):
    r"""
    Adapted from `~transformers.generation_utils.GenerationMixin.sample` to accept a list of input_ids

    - Code source:
        https://github.com/huggingface/transformers/blob/07e3454f034b4889925621e8e3253547d2a04aa7/src/transformers/generation/utils.py#L2454
    - Beam search support:
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py
    """

    # Clear the gpu memory
    torch.cuda.empty_cache()

    num_models = len(models)
    device = models[0].model.device
    batch_size = len(batch[0])
    batch_sample_size = batch_size * num_samples

    for model_i, model in enumerate(models):
        # Initialize each model with the input
        model.set_input(batch[model_i], num_samples, max_length, sample=True)
    
    # Hypotheses on beams across models are always consistent, but one may be shorter than another.
    # We keep track of which model is stalled for any given beam/hypothesis item
    stalled_states = [[False for _ in range(num_models)] for _ in range(batch_sample_size)]
    postfixes = [["".encode('utf-8') for _ in range(num_models)] for _ in range(batch_sample_size)]

    completed = [[] for _ in range(batch_size)]
    continue_search = [True for _ in range(batch_size)]
    step_i = 0

    while any(continue_search):
        paired_outputs = defaultdict(lambda: defaultdict(list)) # each beam item has a list of outputs for each model
        cached_steps = []

        # This stops if our models are at their maximum generation length
        force_stop = False
        for model in models:
            if model.output_ids.shape[1] == max_length:
                force_stop = True

        # Each model still steps individually
        for model_i, model in enumerate(models):

            # Right now we step on all items -- even if model is stalled.
            step_outputs, next_token_scores = model.step()

            # Cache the transformer decoder statement for faster decoding
            cached_steps.append(step_outputs)
            
            # Some debugging prints
            logger.debug(f"STEP {step_i} ({model.model_name}) MODEL {model_i} STALL {stalled_states}")
            for line in model.get_logging_string(step_i, model_i, type="SAMPLE"):
                logger.debug(line)

            for beam_i, beam_expansion in enumerate(next_token_scores):
                paired_outputs[beam_i][model_i] = get_sample_output_extensions(
                    stalled_states[beam_i][model_i],
                    beam_expansion,
                    model.beam_scores[beam_i],
                    model.eos_token_ids[0],
                    device=device,
                    model=model,
                    postfix=postfixes[beam_i][model_i],
                    force_stop = force_stop         
                )


        # All models have stepped. Start search by seeding the heap with the best candidates from each beam
        next_samples = []
        for batch_i in range(batch_size):
            if continue_search[batch_i]:

                # This is our current best hypothesis. We don't want to continue our search if we every pass this score
                next_batch_samples, completed_items = sample_search(
                    batch_offset = batch_i * num_samples,
                    num_samples = num_samples,
                    paired_outputs = paired_outputs,
                    models = models,
                    weights = weights,
                    stalled_states = stalled_states[batch_i * num_samples: (batch_i + 1) * num_samples],
                    max_length = max_length,
                    postfixes=postfixes,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # If we have any items that have been completed, we need to add them to the completed list
                completed[batch_i].extend(completed_items)
                logger.debug(f"BEAM {batch_i} COMPLETED {len(completed[batch_i])} SAMPLES")

                # If we've completed all samples then we quit searching
                if len(completed[batch_i]) == num_samples:
                    continue_search[batch_i] = False

                for beam_i, beam in enumerate(next_batch_samples):
                    candidates = [beam[0].outputs[_][1].token for _ in range(num_models)]
                    candidates_strings = " ||| ".join(candidates)
                    postfix_strings = b"|||".join(beam[2])
                    logger.debug(f"SELECTED {beam_i} {candidates_strings}")
                    logger.debug(f"POSTFIXES: {postfix_strings}")
            else:
                # if this batch item is done, we need to pad the beam with empty items
                # this is due to having the batch_beam_size
                next_batch_beam = get_pad_beams([], models, batch_i, num_samples, weights)

            next_samples.extend(next_batch_samples)

        # after all our batch items have been extended, we'll update the models
        stalled_states, postfixes = update_models_with_beams(
            next_samples,
            models,
            cached_steps,
            last_stalled_states=stalled_states              
            )
        step_i += 1


    outputs = []
    for batch_i, complete in enumerate(completed):
        # TODO: if a hypothesis is incomplete (because max length has been reached, this will error due to the padding values)
        # TODO: fill in the dummy function in utils and call here when applicable

        outputs.append([])
        for sample_i, sample in enumerate(complete):
            logger.debug(f"SAMPLE {sample_i} for BATCH {batch_i}")
            scores = [_ for _ in sample.scores]
            output_tokens = [model.target_tokenizer.convert_ids_to_tokens(sample.output_ids[model_i]) for model_i, model in enumerate(models)]
            combined_score = sample.raw_score()
            out_str = models[0].target_tokenizer.decode(sample.output_ids[0], skip_special_tokens=True)

            for model_i, model in enumerate(models):
                ids = sample.output_ids[model_i]
                tokens = model.target_tokenizer.convert_ids_to_tokens(ids)
                logger.debug(f"MODEL {model_i}")
                logger.debug(f"IDS: {ids}")
                logger.debug(f"TOKS: {tokens}")

            input_ids = [model.input_ids[batch_i].tolist() for model in models if model.is_encoder_decoder]

            outputs[-1].append(
                {
                    "input_ids": input_ids,
                    "sequence": out_str,
                    "scores": scores,
                    "combined_score": combined_score,
                    "token_scores": sample.token_scores,
                    "tokens": output_tokens,
                    "token_ids": sample.output_ids,
                    "weights": weights
                }
            )

    return outputs

def get_sample_output_extensions(
        stalled,
        next_token_scores,
        beam_score,
        eos_token_id,
        device : torch.device = torch.device('cpu'),
        trie: bool = False,
        model: Model = None,
        postfix : str = None,
        force_stop : bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if force_stop:
        # out = next_token_scores + beam_score
        mask = torch.full(next_token_scores.shape, float('-inf')).to(device)
        mask[eos_token_id] = 0
        return next_token_scores + beam_score + mask
        # return [[(next_token_scores + beam_score)[eos_token_id]], torch.tensor([eos_token_id], dtype=torch.long, device=device)]

    if stalled:
        return beam_score.view(1, 1)
    
    # If a trie was constructed, select the top-k tokens that are in the trie (limits sort and search space)
    # if trie:
    #     if postfix != "":
    #         mask = model.trie.search_key_inf_mask(postfix).to(device)
    #         return (next_token_scores + mask + beam_score)

    # need to somehow expand this
    return (next_token_scores + beam_score)












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
        if args.command == "sample":
            return [_.get("sequence", "Error!") for _ in output]
        else:
            return output.get("sequence", "Error!")

def print_output(outputs, args, ostream):
    for o in outputs:
        print(build_output(o, args), file=ostream)


def ensemble_models(args):
    import time
    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    logging.debug(f"Using device: {device}")

    start = time.time()
    models = get_models(args.models, device, args.cache, args.half)
    weights = [w / sum(args.weights) for w in args.weights] if args.weights is not None else [1/len(models) for _ in models]
    if args.trie:
        build_tries(models)
    end = time.time()
    logger.info(f"Time to load models: {end - start}")
    istream = open(args.input, 'r') if args.input else sys.stdin
    ostream = open(args.output, 'w') if args.output else sys.stdout

    batches = batch_generator(istream, args.batch_size, len(models))
    
    start = time.time()
    for i, batch in enumerate(batches):
        # empty cuda cache
        torch.cuda.empty_cache()
        if args.command == 'beam':
            outputs = ensemble_beam_search(
                        batch,
                        models,
                        weights,
                        num_beams = args.num_beams,
                        max_length = args.max_length,
                        trie = args.trie)
        else:
            outputs = ensemble_sample(
                        batch,
                        models,
                        weights,
                        num_samples = args.num_samples,
                        max_length = args.max_length,
                        temperature = args.temperature,
                        top_k = args.top_k,
                        top_p = args.top_p)

        print_output(outputs, args, ostream)

    end = time.time()
    print(f"Time to process: {end - start}", file=sys.stderr)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble models')

    parser.add_argument("--input", '-i', type=str, help='Input file. Defaults to stdin', default=None)
    parser.add_argument("--output", '-o', type=str, help='Output file. Defaults to stdout', default=None)

    parser.add_argument("--models", '-m', type=str, help='Models to ensemble', nargs='+', default=["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"])
    parser.add_argument("--weights", '-w', type=float, help='Weights for each model', nargs='+')

    parser.add_argument("--batch-size", '-bs', type=int, help='Batch size for inference', default=1)
    parser.add_argument("--max-length", '-l', type=int, help='Maximum length of the output', default=256)
    parser.add_argument("--score", '-s', action='store_true', help='Output the score of each model')

    parser.add_argument("--trie", default=False, action='store_true', help='Use trie for finding extensions')
    parser.add_argument("--cross", default=False, action='store_true', help='Build the filter for the cross product of the vocabularies in advance')

    parser.add_argument("--cpu", '-c', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument("--cache", '-a', action='store_true', help='Cache the models')
    parser.add_argument("--half", '-f', action='store_true', help='Use half precision (fp16) for inference')

    parser.add_argument("--debug", '-d', action='store_true', help='Debug mode')

    # Add a beam search subparser
    search = parser.add_subparsers(dest='command', required=True)
    parser_beam = search.add_parser('beam', help='Beam search')
    parser_beam.add_argument("--num-beams", '-b', type=int, help='Number of beams', default=5)

    # Add a sample search subparser
    parser_sample = search.add_parser('sample', help='Sample search')
    parser_sample.add_argument("--num-samples", '-n', type=int, help='Number of samples', default=1)
    parser_sample.add_argument("--temperature", '-t', type=float, help="Temperature for sampling", default=1.0)
    parser_sample.add_argument("--top-k", '-k', type=int, help="Top-k sampling", default=-1)
    parser_sample.add_argument("--top-p", '-p', type=float, help="Top-p sampling", default=1.0)

    args = parser.parse_args()


    if args.debug:
        logging.getLogger("ensembling").setLevel(logging.DEBUG)

    if args.weights is not None:
        assert len(args.models) == len(args.weights), "Number of models and weights must be the same"
        assert all([w >= 0 for w in args.weights]), "Weights must be non-negative"
    assert args.command != "beam" or args.num_beams > 0, "Number of beams must be positive"
    assert args.command != "sample" or args.num_samples > 0, "Number of samples must be positive"
    assert args.batch_size > 0, "Batch size must be positive"

    if args.command == "sample" and not args.trie:
        logger.info("Setting `--trie` to True for sampling")
        args.trie = True

    ensemble_models(args)



    
    

    
    
    
    
