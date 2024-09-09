#!/usr/bin/env python3
import os, sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")

import argparse
from collections import defaultdict
import heapq
import torch

from typing import Optional, Tuple, Union, List, Dict, Any
from torch import nn, LongTensor

import time

from transformers import (
    LogitsProcessor,
    BeamSearchScorer,
    MaxLengthCriteria,
)

from transformers.generation.utils import (
    GenerateBeamOutput, 
    GenerateBeamDecoderOnlyOutput, 
    GenerateBeamEncoderDecoderOutput,
)
from models import get_model_bundle, Bundle

__version__ = "0.1.0"

class BeamState():
    def __init__(self, outputs=None, beam_index=0, weights=None):
        self.outputs = outputs # list of tuples (id, TokenExtension)
        self.beam_index = beam_index
        if weights is None:
            self.weights = [(1.0 / len(outputs)) for _ in range(len(outputs))]
        else:
            self.weights = weights
        self.weighted_score = sum([self.weights[i] * (output[1].score/output[1].hyp_len) for i, output in enumerate(self.outputs)])

    def score(self):
        return self.weighted_score

    def __str__(self):
        out_str = f"STATE({self.beam_index})"
        for output in self.outputs:
            out_str += f" {output[0]}"
        return out_str
    
    # def __hash__(self):
    #     return hash((self.outputs, self.beam_index))

    def __lt__(self, other):
        return self.weighted_score > other.weighted_score
    
    # def __gt__(self, other):
    #     return self.score() > other.score()

    def __eq__(self, other):
        return self.score == other.score

class TokenExtension():
    def __init__(self, score, index, token, hyp_len):
        self.score = score
        self.index = index
        self.token = token
        self.hyp_len = hyp_len

def expand_frontier(bundles, state, paired_outputs, stalled_states):
    beam_i = state.beam_index
    stalls = stalled_states[beam_i]
    neighbors = []
    for model_i, stall in enumerate(stalls):
        add = True
        if not stall:
            outputs = []
            for output_i, state_output in enumerate(state.outputs):
                if model_i == output_i:
                    next_id = state_output[0] + 1
                    if next_id < len(paired_outputs[beam_i][model_i][0]):
                        outputs.append(
                            (next_id, TokenExtension(score=paired_outputs[beam_i][model_i][0][next_id],
                                                    index=paired_outputs[beam_i][model_i][1][next_id],
                                                    token=bundles[model_i].id_to_token(paired_outputs[beam_i][model_i][1][next_id]),
                                                    hyp_len=len(bundles[model_i].decoder_prefixes[beam_i]) + 1))
                        )
                    else:
                        add = False
                else:
                    outputs.append(state_output) #??
            if add:
                neighbors.append(
                    BeamState(outputs=outputs, beam_index=beam_i, weights=state.weights)
                )
    return neighbors

def compatibility(bundles, next_state):
    # 0 means compatible and all finished
    # 1 means compatible
    # -1 means incompatible
    num_models = len(bundles)
    candidate_strings = []
    for i in range(num_models):
        candidate_strings.append(bundles[i].extend_beam_string(next_state.beam_index, next_state.outputs[i][1].index))
    # candidate_strings = [bundles[i].extend_beam_string(next_state.beam_index, next_state.outputs[i][1].index) for i in range(num_models)]
        
    string_lengths = [len(candidate_strings[i]) for i in range(num_models)]

    eos_endings = [bundles[i].is_eos(next_state.outputs[i][1].index) for i in range(num_models)]

    max_length = max(string_lengths)
    min_length = min(string_lengths)

    # if any strings have ended, compatibility must be:
    # - exactly equal for finished strings
    # - substring only for unfinished strings
    if any(eos_endings):
        finished_strings = [candidate_strings[i] for i in range(num_models) if eos_endings[i]]
        unfinished_strings = [candidate_strings[i] for i in range(num_models) if not eos_endings[i]]
        leading_string = finished_strings[0]
        compatibilities = [leading_string == fin for fin in finished_strings] + [leading_string.startswith(ufin) for ufin in unfinished_strings]
        if all(compatibilities):
            if all(eos_endings):
                return 0, None
            else:
                return 1, eos_endings
        else:
            return -1, None

    # otherwise, leading string determines compatibility
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
    
class Hypothesis():
    def __init__(self, output_ids, scores):
        self.output_ids = output_ids
        self.scores = scores

    def score(self):
        return sum(self.scores) / len(self.scores)


@torch.no_grad()
def ensemble_beam_search(
        input: str,
        bundles: List[Bundle],
        weights: Optional[List[float]] = None,
        max_steps: Optional[int] = None,
        num_beams: int = 1,
        debug=False
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    Adapted from `~transformers.generation_utils.GenerationMixin.beam_search` to accept a list of input_ids

    - Code source:
        https://github.com/huggingface/transformers/blob/07e3454f034b4889925621e8e3253547d2a04aa7/src/transformers/generation/utils.py#L2764
    - Beam search support:
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py

    TODO:
    - [ ] Generalize to n models  
    """

    num_models = len(bundles)
    device = bundles[0].model.device

    batch_size = 1  # len(beam_scorer._beam_hyps)

    start_time = time.time()
    for bundle in bundles:
        # Initialize each model with the input
        # TODO: maybe this is where the first decoder token should also be set?
        bundle.set_input(input, num_beams=num_beams, max_length=max_steps)

    # Hypotheses on beams across models are always consistent, but one may be shorter than another.
    # 0: model 0 has a shorter surface string
    # 1: model 1 has a shorter surface string
    # 2: the models are in sync
    # Calls to a model's topk() are not allowed to select from beam items where that model is ahead
    stalled_states = [[False for _ in range(num_models)] for __ in range(num_beams)]

    beam_completed = []  # contains completed sentences

    """
    # Models: 0 (TM) and 1 (LLM)
    BEAM 0: [Life is like a box of], [Life is like a box of]    # -> sync state 2
    BEAM 1: [Life is like a container], [Life is like a cont]   # -> sync state 1 (model 1 is shorter)
    BEAM 2: [Life is similar to a], [Life is similar to a box]  # -> sync state 0 (model 0 is shorter)
    """
    for step_i in range(1, max_steps + 1): # could change to a while true or count the tokens each model has produced
        # A heap that will store all beam candidates, used to form the next beam.
        # This will include single-model advances (where one model is behind and is trying to catch up)
        # and paired-model advances (where both models advanced a compatible token).
        candidates = [[] for _ in range(num_beams)] # priority queue/heap
        visited = set() # keeping track of which states get pushed so we don't push the same one many times

        paired_outputs = defaultdict(lambda: defaultdict(list)) # the list of outputs from each model for each beam
        cached_steps = []

        # Take the next step of each model
        for model_i, bundle in enumerate(bundles):
            step_outputs, next_token_scores = bundle.step(step_i) #perhaps some room for efficiency here--maybe move below if statement?
            sequence_scores = next_token_scores + bundle.beam_scores[:, None].expand_as(next_token_scores) # k x V

            cached_steps.append(step_outputs)

            # do top-k selection where the model is behind or synced
            if args.debug:
                print("STEP", type(bundle.model), step_i, "MODEL", model_i, "STALL", stalled_states)
                bundle.print_beam(model_i, step_i)

            for beam_i, beam_expansion in enumerate(sequence_scores):
                if stalled_states[beam_i][model_i]:
                    paired_outputs[beam_i][model_i] = torch.tensor([[0], [bundle.tokenizer.pad_token_id]], device=device)
                else:
                    paired_outputs[beam_i][model_i] = torch.sort(beam_expansion, descending=True)

        # All models have stepped. Start search by seeding the heap with the best candidates from each beam
        for beam_i in range(num_beams):
            next_state = BeamState(
                # 0 signifies that this is the 0th index of the beam item
                # paired_outputs --> beam_i x model_i x scores x token_ids
                outputs = [
                    (0, TokenExtension(score=paired_outputs[beam_i][model_i][0][0],
                                       index=paired_outputs[beam_i][model_i][1][0],
                                       token=bundles[model_i].id_to_token(paired_outputs[beam_i][model_i][1][0]),
                                       hyp_len=len(bundles[model_i].decoder_prefixes[beam_i]) + 1))
                    for model_i in range(num_models)
                ],
                beam_index=beam_i,
                weights=weights
            )
            heapq.heappush(candidates[beam_i], next_state)
            visited.add(str(next_state))

        # Now, we will explore the heap to find the best candidates
        next_beam = []
        while len(next_beam) < num_beams and len(beam_completed) < num_beams and sum(len(cand) for cand in candidates) > 0:
            for candidate_heap in candidates:
                for chance in range(num_beams):
                    next_state = heapq.heappop(candidate_heap)

                    # add neighbors regardless of compatibility
                    for neighbor in expand_frontier(bundles, next_state, paired_outputs, stalled_states):
                        if str(neighbor) not in visited:
                            visited.add(str(neighbor))
                            heapq.heappush(candidate_heap, neighbor)

                    compat_code, next_stall_states = compatibility(bundles, next_state)
                    if compat_code == 0:
                        # all models have terminated with eos
                        beam_completed.append(Hypothesis(
                            output_ids = [bundles[i].output_ids[next_state.beam_index] for i in range(num_models)],
                            scores = [next_state.outputs[i][1].score for i in range(num_models)],
                        ))
                    elif compat_code == 1:
                        next_beam.append((next_state, next_stall_states))
                if step_i == 1:
                    break

        # trim beam:
        next_beam = sorted(next_beam, key=lambda x: x[0].score(), reverse=True)[:num_beams]

        if args.debug:
            print("I COUNT", len(beam_completed), "COMPLETED")
        if len(beam_completed) >= num_beams:  # you should break if beam_completed[0] < next_beam[0]
            if args.debug:
                print("DONE")
            break

        # end early if the best candidate is worse than the best completed beam
        if len(beam_completed) > 0 and next_beam[0][0].score() < beam_completed[0].score():
            break

        for i in range(num_beams):
            candidates = [next_beam[i][0].outputs[_][1].token for _ in range(num_models)]
            candidates_strings = " ||| ".join(candidates)
            if args.debug:
                print("SELECTED", i, candidates_strings)        
        # Now, go through and update each model with its selection.
        # There is a constraint on updating: a model can only update beam items
        # where the sync state is 2 (both models were already in sync) or where
        # the sync state is the model number (that model was behind).
        if args.debug:
            print(f"VISITED_STATES: {len(visited)}")
        stalled_states = [[] for _ in range(num_beams)] # beam indexed
        for i, bundle in enumerate(bundles):
            beam_indices = [next_beam[j][0].beam_index for j in range(len(next_beam))]
            beam_tokens = [next_beam[j][0].outputs[i][1].index.item() for j in range(len(next_beam))]
            beam_scores = [next_beam[j][0].outputs[i][1].score for j in range(len(next_beam))]
            update_mask = [next_beam[j][1][i] for j in range(len(next_beam))]
            for j in range(num_beams):
                stalled_states[j].append(update_mask[j])
            if args.debug:
                print("MODEL", i, "STEP", step_i, "UPDATE", beam_indices, beam_tokens, update_mask)
            bundle.update(beam_indices, beam_tokens, beam_scores, debug=args.debug, step_outputs=cached_steps[i])
            # bundle.update(beam_indices, beam_tokens, beam_scores, debug=args.debug, step_outputs=None)


    # One case where there aren't completed beams is if our max_steps was too short. We must return the beams anyway
    while len(beam_completed) < num_beams:
        best_cand = next_beam.pop(0)
        beam_completed.append(
                Hypothesis(
                    output_ids = [bundles[i].output_ids[best_cand[0].beam_index] for i in range(num_models)],
                    scores = [best_cand[0].outputs[i][1].score for i in range(num_models)],
                )
        )
 
    sorted_completions = sorted(beam_completed, key=lambda x: x.score(), reverse=True)

    for i, completed in enumerate(beam_completed):
        model0_str = bundles[0].tokenizer.decode(completed.output_ids[0], skip_special_tokens=True)
        scores = completed.scores
        pair_score = completed.score()
        if args.debug:
            print(f"COMPLETED {i}:", model0_str, scores, pair_score)

    best_beam = sorted_completions[0]
    output_str = bundles[0].tokenizer.decode(best_beam.output_ids[0], skip_special_tokens=True)
    return output_str, best_beam.score(), time.time()-start_time


class RandomNoiseLogitsProcessor(LogitsProcessor):
    def __init__(self, noise):
        self.noise = noise

    def __call__(self, 
                 input_ids: LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
            return scores + torch.randn_like(scores) * self.noise


def main(args):

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.debug(f"Using device: {device}")

    models = []
    for model_name in args.model_names:
        models.append(get_model_bundle(model_name, target_language=args.target_lang, device=device, cache=args.cache))

    if args.noise is not None:
        models[0].logits_processor.append(
            RandomNoiseLogitsProcessor(args.noise)
        )

    weights = [_ / sum(args.weights) for _ in args.weights] if args.weights is not None else None
    if weights is not None:
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

    # istream = ["An activist supporter of Jean-Pierre Chevénement in 2002, he later supported Dominique de Villepin in the district from 2010 to 2011."]

    # input_source = ["Between the early 1970s, when the Boeing 747 jumbo defined modern long-haul travel, and the turn of the century, the weight of the average American 40- to 49-year-old male increased by 10 per cent, according to U.S. Health Department Data."]

    # input_source = ['"It really comes down to providing flexibility to airlines and allowing them to do the things that they believe they need to do to be successful," said Boeing cabins expert Kent Craver.']

    # input_source = ["This is a test.", 
    #                 "this is another test, but it's too long so the model is gonna get stuck before it's able to finish and it won't have enough tokens to generate the eos one."]
    # input_source = [
    #     "How the back of the plane is laid out - particularly whether seating is 9 or 10 abreast - is central to the economic performance claims being made for new \"mini-jumbo\" jet designs."
    # ]
    # input_source = [
    #     "Jet makers feud over seat width with big orders at stake",
    #     "A row has flared up between leading plane makers over the width of tourist-class seats on long-distance flights, setting the tone for a bitter confrontation at this month's Dubai Airshow.",
    #     "The dispute focuses on the width of seats provided on long-haul flights for economy passengers - not always the ones most courted by airlines, but whose allocated space holds the key to efficiency claims for the latest jets offered by Airbus SAS and Boeing Co.",
    #     "Airbus this week called for an industry standard that would provide for a seat at least 18 inches (46 cm) wide in economy cabins, but its U.S. arch-rival Boeing says it should be for airlines to decide.",
    #     "The dispute comes as plane makers vie to sell ever-larger versions of their twin-engined long-distance aircraft, with potentially record orders expected at the November 17-21 event.",
    #     "How the back of the plane is laid out - particularly whether seating is 9 or 10 abreast - is central to the economic performance claims being made for new \"mini-jumbo\" jet designs.",
    #     "Boeing says its revamped \"777X\" will hold 406 people based on economy seats more than 17 inches wide and set out 10 in each row.",
    #     "Airbus says the competing version of its A350 will carry 350 people in 18-inch-wide economy seat laid out 9 abreast.",
    #     "Plane giants often trade blows on technical matters through advertising in the trade press.",
    #     "Now, Airbus is appealing directly to the public ahead of the Dubai Airshow, where the 777X is expected to dominate with more than 100 orders."
    # ]
    istream = sys.stdin if args.input is None else open(args.input, "r")

    for line in istream:
    # for line in input_source:
        line = line.rstrip()

        # normally you would now call beam search, but we need to implement it
        outputs = ensemble_beam_search(line, 
                                       models, 
                                       num_beams=args.num_beams, 
                                       max_steps=args.max_steps, 
                                       weights=weights,
                                       debug=args.debug)
        output_string = build_output_string(args, outputs)
        print(output_string)

def build_output_string(args, output):
    out = ""
    if args.debug:
        out += f"PPL: {output[1]}\t"
    if args.time:
        out += f"TIME: {output[2]}\t"
    out += f"{output[0]}"
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=None, type=str, help="Input stream")
    parser.add_argument("--model-names", "-m", type=str, nargs="+", default=["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"], help="Model names")
    parser.add_argument("--weights", default=None, type=float, nargs="+", help="Weights for each model")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--source-lang", "-s", type=str, default="en", help="Source language")
    parser.add_argument("--target-lang", "-t", type=str, default="fr", help="Target language")
    parser.add_argument("--num-beams", "-b", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-steps", "-l", type=int, default=30, help="Maximum number of output tokens")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--cache", default=False, action='store_true')
    parser.add_argument("--time", default=False, action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("ensembling").setLevel(logging.DEBUG)

    main(args)