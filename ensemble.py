#!/usr/bin/env python3

import argparse
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

STEP = 0


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
                next_token_logits = step_outputs.logits[:, -1, :]
                # print("* OUTPUTS.LOGITS", next_token_logits.shape)

                forced_tokens = torch.ones((num_beams, 1), device=device, dtype=torch.long) * model.bos_force_token
                model_output_ids = torch.cat([model_output_ids, forced_tokens], dim=-1)

                # Initialize models, including running over force BOS tokens
            # These store individual models' tokenized outputs
            self.model_output_ids.append(model_output_ids)

    def step(self):
        """
        Take a step of the ensembled model and fill a new beam. There are two kinds of steps, 
        corresponding to the state of each beam item:
        - **Sychronized**. In this case, all models exactly agree on the output.
          Each model takes a step, and the outputs are compared and merged.
        - **Unsynchronized**. In this case, the models have consistent outputs, but might
          have generated different lengths. In this case, only the models that are behind are
          allowed to take a step. From its output, we choose only items that are consistent
          with the already-generated string.
        """
        # SYNCHRONIZED STEP
        # For every model, take a step from its current states.
        # Then filter its rows to only include those that are synchronized.
        # Apply top-k selection to select the top-k candidate items for that model

        for model, output_ids in zip(self.models, self.model_output_ids):
            model_inputs = model.prepare_inputs_for_generation(output_ids)

            step_outputs = model.step(model_inputs)

            # Step
            step_outputs = model.step(model_inputs)
            next_token_logits = step_outputs.logits[:, -1, :]
            # Massage the logits. This is how prefix decoding is enforced.
            next_token_logits = model.logits_processor(model_output_ids, next_token_logits)
            next_token_scores = nn.functional.softmax(
                next_token_logits, dim=-1
            )


class BeamItem:
    """
    Represents beam items.
    """
    def __init__(self, index, token, score, model_no=None):
        self.index = index  # the beam index it extends
        self.token = token  # the token itself
        self.score = score  # the score of the token
        self.model_no = model_no  # the model that generated this token (None: synced)

    def __str__(self):
        if self.model_no:
            return f"ITEM({self.index}, {self.token}, {self.score} #{self.model_no})"
        else:
            return f"ITEM({self.index}, {self.token}, {self.score})"

    def __lt__(self, other):
        return self.score < other.score


class BeamItemPair:
    """
    Represents a pair of beam items.
    """
    def __init__(
            self,
            cand0: BeamItem, 
            cand0_rank: int,
            cand1: BeamItem, 
            cand1_rank: int,
            synced=False):
        self.cand0 = cand0
        self.cand0_rank = cand0_rank
        self.cand1 = cand1
        self.cand1_rank = cand1_rank
        self.synced = synced

    def score(self):
        """
        Returns the interpolated score of the pair.
        """
        return (self.cand0.score + self.cand1.score) / 2

    def __str__(self):
        return f"PAIR({self.cand0}, {self.cand1})"

    def __lt__(self, other):
        return self.cand0.score < other.cand1.score


def get_sync_mask(self):
    """
    Return a mask denoting items in the beam that are synchronized.
    """
    return self.synchronized

@torch.no_grad()
def ensemble_beam_search(
        input: str,
        bundles: List[Bundle],
        max_length: Optional[int] = None,
        num_beams: int = 1,
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
        bundle.set_input(input, num_beams=num_beams, max_length=max_length)

    # beam = EnsembleBeam()

    def compatible(cand1, cand2) -> bool:
        cand1_str = bundles[0].get_hyp_str(cand1.index, cand1.token)
        cand2_str = bundles[1].get_hyp_str(cand2.index, cand2.token)

        if len(cand1_str) < len(cand2_str):
            result = cand2_str.startswith(cand1_str), -1
        elif len(cand1_str) > len(cand2_str):
            result = cand1_str.startswith(cand2_str), 1
        else:
            result = cand1_str == cand2_str, 0

        # print("CMP", cand1_str, " ||| ", cand2_str, "=", result)

        return result
    
    # Values:
    # 0: model 0 is ahead
    # 1: model 1 is ahead
    # 2: models are in sync
    sync_states = torch.tensor([[2 for _ in range(num_beams)] for _ in range(batch_size)], dtype=torch.short, device=device)

    for step_i in range(1, args.max_output_tokens + 1):

        # TODO: add preprocessing abstraction

        # transform each row of output_ids into tokens and print
 
        candidates = [[] for _ in range(num_models)]

        # Take the next step of each model
        for model_i, bundle in enumerate(bundles):

            step_outputs, next_token_scores = bundle.step(step_i)
            sequence_scores = next_token_scores + bundle.beam_scores[:, None].expand_as(next_token_scores)

            # do top-k selection where the model is behind or synced
            print("STEP", step_i, "MODEL", model_i, "SYNC", sync_states, (sync_states == 2) | (sync_states == model_i))
            bundle.print_beam(model_i, step_i)
            if torch.any(region := ((sync_states == 2) | (sync_states == model_i))):
                next_indices, next_tokens, next_token_scores = bundle.topk(sequence_scores, region)

                # create a candidate out of it
                for index, token, score in zip(next_indices[0], next_tokens[0], next_token_scores[0]):
                    # print("ITEM", index, token, score)
                    cand = BeamItem(index, token, score, model_no=model_i)
                    candidates[model_i].append(cand)

            # TODO: we may want to make two top-k calls per model, separating synced from non-synced rows for each model

            # token_str = "  ".join([f"{str(int(i))}/{bundle.tokenizer.decode(t, skip_special_tokens=False)}/{t}" for i, t in zip(next_indices[0, :], next_tokens[0, :])])
            # print("TOPK", model_i, token_str, next_token_scores)
            print("TOPK", model_i)
            for i, cand in enumerate(candidates[model_i]):
                # get the token str from the cand.token using the tokenizer
                token_str = bundle.tokenizer.convert_ids_to_tokens([cand.token])[0]
                print("->", i, cand, token_str)

            # topk on items where this model is behind
            # topk = bundle.topk(outputs, self.get_behind_mask(model_i))
            # for item in topk:
            #     # TODO: actually take the step, and push the item on the candidates list
            #     cand = Candidate(item)
            #     heapq.heappush(candidates[model_i], cand)

            # for cand in candidates[model_i]:
                # concatenate the token to the beam
                # print("MODEL", model_i, "CAND", bundle.get_hyp_str(cand.index, cand.token))

        # Now we have {topk} x {topk} items, which we need to merge into a single list that will
        # become the new beam. However, (a) not all items are compatible and (b) we'd like to
        # avoid quadratic exploration, while still ensuring that we explor a diverse set of 
        # candidate pairs. To do this, we will walk the 2d grid, pushing joined candidates onto
        # a heap. At the end, the heap will contain the topk items.
        #
        # Note that this is akin to k-best extraction with a monotonic combination function, such
        # as for binarized parsing.

        # TODO: seed beam with the complete diagonal, and ensure each item is used only once
        q = [ BeamItemPair( candidates[0][0], 0, candidates[1][0], 0 ) ]
        beam_selection = []  # contains selected items
        beam_completed = []  # contains completed sentences
        seen = set()  # popped items that shouldn't be appended again
        while len(q) > 0 and len(beam_selection) < num_beams:
            pair = heapq.heappop(q)
            seen.add((pair.cand0_rank, pair.cand1_rank))
            cand0 = pair.cand0
            cand1 = pair.cand1
            is_compat, direction = compatible(cand0, cand1)
            cand0_str = bundles[0].get_hyp_str(cand0.index, cand0.token)
            cand1_str = bundles[1].get_hyp_str(cand1.index, cand1.token)
            # print(step_i, "POP", pair.score(), pair.cand0_rank, pair.cand1_rank, cand0_str, "<>", cand1_str, is_compat)

            # add successor item
            ranks = (pair.cand0_rank + 1, pair.cand1_rank + 1)
            if ranks[0] < len(candidates[0]) and ranks[1] < len(candidates[1]) and ranks not in seen:
                heapq.heappush(q, BeamItemPair(candidates[0][ranks[0]], ranks[0], candidates[1][ranks[1]], ranks[1], synced=direction==2))

            if is_compat:
                if cand0.token == bundles[0].eos_token_id and cand1.token == bundles[1].eos_token_id:
                    beam_completed.append((cand0, cand1))
                else:
                    beam_selection.append((cand0, cand1))
            else:
                # add successors
                ranks = (pair.cand0_rank + 1, pair.cand1_rank)
                if ranks[0] < len(candidates[0]) and ranks not in seen:
                    extend_0 = BeamItemPair(candidates[0][ranks[0]], ranks[0], cand1, ranks[1])
                    # print("EXTEND", pair.cand1_rank + 1, len(candidates[0]), extend_2)
                    heapq.heappush(q, extend_0)

                ranks = (pair.cand0_rank, pair.cand1_rank + 1)
                if ranks[1] < len(candidates[1]) and ranks not in seen:
                    extend_1 = BeamItemPair(cand0, ranks[0], candidates[1][ranks[1]], ranks[1])
                    # print("EXTEND", pair.cand2_rank + 1, len(candidates[1]), extend_1)
                    heapq.heappush(q, extend_1)


        if len(beam_completed):
            print("DONEZO")
            break

        # Now enumerate the outputs and use them to update the beams in each sub-model
        if len(beam_selection) < num_beams:
            print("* FATAL: not enough candidates", len(beam_selection), "need", num_beams)
            sys.exit(1)

        beam_selection = beam_selection[:num_beams]
        for i in range(num_beams):
            cand0, cand1 = beam_selection[i]
            cand0_str = bundles[0].get_hyp_str(cand0.index, cand0.token)
            cand1_str = bundles[1].get_hyp_str(cand1.index, cand1.token)            
            print("SELECTED", i, cand0, cand0_str, cand1, cand1_str)
        
        for i, bundle in enumerate(bundles):
            beam_tokens = [beam_selection[j][i].token for j in range(len(beam_selection))]
            beam_indices = [beam_selection[j][i].index for j in range(len(beam_selection))]
            beam_scores = [beam_selection[j][i].score for j in range(len(beam_selection))]

            print("MODEL", i, "STEP", step_i, "UPDATE", beam_tokens, beam_indices)
            bundle.update(beam_tokens, beam_indices, beam_scores)

            # next_token_scores = torch.tensor([selected[j][1].score for j in range(len(selected))], dtype=torch.float, device=device).unsqueeze(0)
            # next_tokens = torch.tensor([selected[j][i].token for j in range(len(selected))], dtype=torch.long, device=device).unsqueeze(0)
            # next_indices = torch.tensor([selected[j][i].index for j in range(len(selected))], dtype=torch.long, device=device).unsqueeze(0)
            # print("MODEL", i, next_token_scores, next_tokens, next_indices)

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder, model_inputs=model_inputs
            # )
            # if model_kwargs["past_key_values"] is not None:
            #     model_kwargs["past_key_values"] = self._temporary_reorder_cache(
            #         model_kwargs["past_key_values"], beam_idx
            #     )

            # if return_dict_in_generate and output_scores:
            #     beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # scores should be the distribution over the next token for each beam item
            # That's a lot of data to keep track of!

    for bundle in bundles:
        sequence_outputs = bundle.finalize()
    # for i, bundle in enumerate(bundles):
    #     sequence_outputs.append(
    #         beam_scorer.finalize(
    #             bundle.output_ids,
    #             bundle.beam_scores,
    #             None,
    #             None,
    #             max_length=stopping_criteria.max_length,
    #             pad_token_id=bundle.pad_token_id,
    #             eos_token_id=bundle.eos_token_id,
    #             beam_indices=bundle.beam_indices,
    #             decoder_prompt_len=bundle.decoder_prompt_len,
    #         )
    #     )

    return sequence_outputs["sequences"]


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

    for line in sys.stdin:
        line = line.rstrip()

        # normally you would now call beam search, but we need to implement it
        outputs = ensemble_beam_search(line, models, num_beams=args.num_beams, max_length=args.max_output_tokens)

        # decode with the combined vocabulary
        result = models[0].tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", "-m", type=str, nargs="+", default=["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"], help="Model names")
    parser.add_argument("--target-lang", "-t", type=str, default="fr", help="Target language")
    parser.add_argument("--num-beams", "-b", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-output-tokens", "-l", type=int, default=30, help="Maximum number of output tokens")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    main(args)