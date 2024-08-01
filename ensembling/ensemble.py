#!/usr/bin/env python3

import argparse
from collections import defaultdict
import heapq
import sys
import torch

from typing import Optional, Tuple, Union, List, Dict, Any
from torch import nn, LongTensor

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


class EnsembleBeam:
    def __init__(self, models, batch_size, num_beams, target_language, device):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.target_language = target_language
        self.device = device

        self.synchronized = [ [ True for _ in range(num_beams) ] for _ in range(batch_size)]

        self.output_strings = ["" for _ in range(num_beams)]

        model_beam_scores = []
        for _ in models:
            # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
            # of the first beam are considered to avoid sampling the exact same tokens across all beams.
            model_beam_scores.append(torch.zeros((batch_size, num_beams), dtype=torch.float, device=device))
            model_beam_scores[-1][:, 1:] = -1e9
            model_beam_scores[-1] = model_beam_scores[-1].view((batch_size * num_beams,))

        self.model_beam_scores = model_beam_scores

        """
        Advance a model just beyond generation of its BOS tokens.
        Calling this on each model ensures that they are all in a position
        to advance freely.
        """
        self.model_output_ids = []
        for model in models:
            model_output_ids = torch.ones((num_beams, 1), device=device, dtype=torch.long)
            model_output_ids = model_output_ids * model.model.config.decoder_start_token_id

            # step through the forced BOS token
            if model.bos_force_token:
                model_inputs = model.prepare_inputs_for_generation(model_output_ids)

                # Step
                step_outputs = model.step(model_inputs)
                # print("* OUTPUTS.LOGITS", next_token_logits.shape)

                forced_tokens = torch.ones((num_beams, 1), device=device, dtype=torch.long) * model.bos_force_token
                model_output_ids = torch.cat([model_output_ids, forced_tokens], dim=-1)

                # Initialize models, including running over force BOS tokens
            # These store individual models' tokenized outputs
            self.model_output_ids.append(model_output_ids)


class BeamItem:
    """
    Represents beam items.
    """
    def __init__(self, model_no, index, token, score):
        self.model_no = model_no  # the model that generated this token (None: synced)
        self.index = index  # the beam index it extends
        self.token = token  # the token itself
        self.score = score  # the score of the token

    def __str__(self):
        return f"ITEM({self.index}, {self.token}, {self.score} #{self.model_no})"


class BeamItemPair:
    """
    Represents a pair of beam items.
    """
    def __init__(
            self,
            cand0: BeamItem,
            cand1: BeamItem,
            sync_state=None):
        """
        """
        self.cand0 = cand0
        self.cand1 = cand1
        self.sync_state = sync_state

    @property
    def token0(self):
        return self.cand0.token
    
    @property
    def token1(self):
        return self.cand1.token

    @property
    def score0(self):
        return self.cand0.score
    
    @property
    def score1(self):
        return self.cand1.score
    
    @property
    def beam_index(self):
        return self.cand0.index

    def __getitem__(self, model_i):
        """
        Return the beam item for the given model.
        """
        return self.cand0 if model_i == 0 else self.cand1

    def __setitem__(self, model_i, value):
        """
        Set the beam item for the given model.
        """
        if model_i == 0:
            self.cand0 = value
        else:
            self.cand1 = value

    def score(self):
        """
        Returns the interpolated score of the pair.
        """
        return (self.score0 + self.score1) / 2

    def __str__(self):
        ensemble_score = self.score()
        return f"PAIR({ensemble_score:.3f} SYNC{self.sync_state} {self.beam_index} -> {self.token0} ({self.score0:.3f}) / {self.token1} ({self.score1:.3f}))"

    def __lt__(self, other):
        return self.score() > other.score()

class BeamState():
    def __init__(self, outputs=[], beam_index=0):
        self.outputs = outputs
        self.beam_index = beam_index

    def score(self):
        return sum([output[1].score for output in self.outputs])

    def __str__(self):
        return f"STATE({self.beam_index} {self.outputs})"

    def __lt__(self, other):
        return self.score() > other.score()

class TokenExtension():
    def __init__(self, score, index, token):
        self.score = score
        self.index = index
        self.token = token

def expand_frontier(bundles, state, paired_outputs, stalled_states):
    beam_i = state.beam_index
    stalls = stalled_states[beam_i]
    neighbors = []
    for model_i, stall in enumerate(stalls):
        if not stall:
            outputs = []
            for output_i, state_output in enumerate(state.outputs):
                if model_i == output_i:
                    next_id = state_output[0] + 1
                    outputs.append(
                        (next_id, TokenExtension(score=paired_outputs[beam_i][model_i][0][next_id],
                                                 index=paired_outputs[beam_i][model_i][1][next_id],
                                                 token=bundles[model_i].id_to_token(paired_outputs[beam_i][model_i][1][next_id])))
                    )
                else:
                    outputs.append(state_output)
            neighbors.append(
                BeamState(outputs=outputs, beam_index=beam_i)
            )
    return neighbors

def compatibility(bundles, next_state):
    # 0 means compatible and all finished
    # 1 means compatible
    # -1 means incompatible
    num_models = len(bundles)
    candidate_strings = [bundles[i].get_surface_str(next_state.beam_index, next_state.outputs[i][1].index) for i in range(num_models)]
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
            return 0, eos_endings
        else:
            return -1, None

    # otherwise, leading string determines compatibility
    leading_string = candidate_strings[string_lengths.index(max_length)]
    compatibilities = [leading_string.startswith(candidate_strings[i]) for i in range(num_models)]
    if all(compatibilities):
        if all(eos_endings):
            return 0, None
        else:
            if max_length == min_length:
                return 1, [False for _ in range(num_models)]
            else:
                return 1, [l == max_length for l in string_lengths]
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
    for step_i in range(1, max_steps + 1):
        # A heap that will store all beam candidates, used to form the next beam.
        # This will include single-model advances (where one model is behind and is trying to catch up)
        # and paired-model advances (where both models advanced a compatible token).
        candidates = []
        paired_outputs = defaultdict(lambda: defaultdict(list))

        # Take the next step of each model
        for model_i, bundle in enumerate(bundles):
            step_outputs, next_token_scores = bundle.step(step_i) #perhaps some room for efficiency here--maybe move below if statement?
            sequence_scores = next_token_scores + bundle.beam_scores[:, None].expand_as(next_token_scores)

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
                                       token=bundles[model_i].id_to_token(paired_outputs[beam_i][model_i][1][0])))
                    for model_i in range(num_models)
                ],
                beam_index=beam_i
            )
            heapq.heappush(candidates, next_state)

        # Now, we will explore the heap to find the best candidates
        next_beam = []
        while len(next_beam) < num_beams and len(candidates):
            next_state = heapq.heappop(candidates)

            # add neighbors regardless of compatibility
            for neighbor in expand_frontier(bundles, next_state, paired_outputs, stalled_states):
                heapq.heappush(candidates, neighbor)

            compat_code, next_stall_states = compatibility(bundles, next_state)
            if compat_code == 0:
                # all models have terminated with eos
                beam_completed.append(Hypothesis(
                    output_ids = [bundles[i].output_ids[next_state.beam_index] for i in range(num_models)],
                    scores = [next_state.outputs[i][1].score for i in range(num_models)],
                ))
                if len(beam_completed) == num_beams:
                    break
            elif compat_code == 1:
                next_beam.append((next_state, next_stall_states))

        # Now, update the beams with the next beam

        # Now we have {topk} x {topk} items, which we need to merge into a single list that will
        # become the new beam. However, (a) not all items are compatible and (b) we'd like to
        # avoid quadratic exploration, while still ensuring that we explor a diverse set of 
        # candidate pairs. To do this, we will walk the 2d grid, pushing joined candidates onto
        # a heap. At the end, the heap will contain the topk items.
        #
        # Note that this is akin to k-best extraction with a monotonic combination function, such
        # as for binarized parsing.


        # PICKUP figuring out when you're done
        if args.debug:
            print("I COUNT", len(beam_completed), "COMPLETED")
        if len(beam_completed) == num_beams:
            if args.debug:
                print("DONE")
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
        for i, bundle in enumerate(bundles):
            beam_indices = [next_beam[j][0].beam_index for j in range(len(next_beam))]
            beam_tokens = [next_beam[j][0].outputs[i][1].index.item() for j in range(len(next_beam))]
            beam_scores = [next_beam[j][0].outputs[i][1].score for j in range(len(next_beam))]
            update_mask = [next_beam[j][1][i] for j in range(len(next_beam))]

            if args.debug:
                print("MODEL", i, "STEP", step_i, "UPDATE", beam_indices, beam_tokens, update_mask)
            bundle.update(beam_indices, beam_tokens, beam_scores, debug=args.debug)

        # update sync states
        # sync_states = torch.tensor([beam_selection[j].sync_state for j in range(len(beam_selection))], dtype=torch.short, device=device)

    # One case where there aren't completed beams is if our max_steps was too short. We must return the beams anyway
    while len(beam_completed) != num_beams:
        best_cand = next_beam.pop(0)
        beam_completed.append(
                Hypothesis(
                    output_ids = [bundles[i].output_ids[best_cand[0].beam_index] for i in range(num_models)],
                    scores = [best_cand[0].outputs[i][1].score for i in range(num_models)],
                )
        )
 
    sorted_completions = sorted(beam_completed, key=lambda x: x.score(), reverse=True)

    for i, completed in enumerate(beam_completed):
        # model0_ids, model0_score, model1_ids, model1_score, pair = completed
        # model_str = bundles[0].get_surface_str(completed.beam_index, completed.outputs[0][1].index)
        model0_str = bundles[0].tokenizer.decode(completed.output_ids[0], skip_special_tokens=True)
        scores = completed.scores
        pair_score = completed.score()
        if args.debug:
            print(f"COMPLETED {i}:", model0_str, scores, pair_score)

    best_beam = sorted_completions[0]
    output_str = bundles[0].tokenizer.decode(best_beam.output_ids[0], skip_special_tokens=True)
    return output_str, best_beam.score()


def is_compatible(cand0_str, cand1_str):
    """
    Determines whether two strings are compatible.
    If so, the sync state is set to mark the string relationship.

    :return: A tuple of (is_compatible, new_sync_state)
    """
    if cand0_str == cand1_str:
        is_compat = True
        new_sync_state = 2
    elif cand0_str.startswith(cand1_str):
        is_compat = True
        new_sync_state = 1
    elif cand1_str.startswith(cand0_str):
        is_compat = True
        new_sync_state = 0
    else:
        is_compat = False
        new_sync_state = None

    return is_compat, new_sync_state


class RandomNoiseLogitsProcessor(LogitsProcessor):
    def __init__(self, noise):
        self.noise = noise

    def __call__(self, 
                 input_ids: LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
            return scores + torch.randn_like(scores) * self.noise


def main(args):

    models = []
    for model_name in args.model_names:
        models.append(get_model_bundle(model_name, target_language=args.target_lang))

    if args.noise is not None:
        models[0].logits_processor.append(
            RandomNoiseLogitsProcessor(args.noise)
        )

    # input_source = ["This is a test.", 
                    # "this is another test, but it's too long so the model is gonna get stuck before it's able to finish and it won't have enough tokens to generate the eos one."]

    for line in sys.stdin:
    # for line in input_source:
        line = line.rstrip()

        # normally you would now call beam search, but we need to implement it
        outputs = ensemble_beam_search(line, models, num_beams=args.num_beams, max_steps=args.max_steps, debug=args.debug)
        if args.debug:
            print(outputs[0], outputs[1])
        else:
            print(outputs[0])

        # decode with the combined vocabulary
        # result = models[0].tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", "-m", type=str, nargs="+", default=["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"], help="Model names")
    parser.add_argument("--source-lang", "-s", type=str, default="en", help="Source language")
    parser.add_argument("--target-lang", "-t", type=str, default="fr", help="Target language")
    parser.add_argument("--num-beams", "-b", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-steps", "-l", type=int, default=30, help="Maximum number of output tokens")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    main(args)