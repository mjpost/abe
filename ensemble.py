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
            beam_index,
            token0,
            score0,
            token1,
            score1,
            sync_state=None):
        """
        """
        self.beam_index = beam_index
        self.token0 = token0
        self.score0 = score0
        self.token1 = token1
        self.score1 = score1
        self.sync_state = sync_state

    def score(self):
        """
        Returns the interpolated score of the pair.
        """
        return (self.score0 + self.score1) / 2

    def __str__(self):
        return f"PAIR({self.index} -> {self.token0}, {self.score0} / {self.token1}, {self.score1})"

    def __lt__(self, other):
        return self.score() < other.score()


class PairedTopk:
    """
    Represents a set of top-k items for each model.
    """
    def __init__(self):
        self.items = {}

    def add(self, model_i, item: BeamItem):
        """
        Append an item to the top-k list.
        """
        beam_i = int(item.index)
        if beam_i not in self.items:
            self.items[beam_i] = [[], []]  # a list of extensions for this beam item for both models

        self.items[beam_i][model_i].append(item)

    def get_beams(self):
        """
        Return the list of beam indices.
        """
        return [key for key in self.items.keys() if len(self.items[key][0]) > 0 and len(self.items[key][1]) > 0]

    def __getitem__(self, index):
        return self.items[index]

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
    
    # Hypotheses on beams across models are always consistent, but one may be shorter than another.
    # 0: model 0 has a shorter surface string
    # 1: model 1 has a shorter surface string
    # 2: the models are in sync
    # Calls to a model's topk() are not allowed to select from beam items where that model is ahead
    sync_states = torch.tensor([2 for _ in range(num_beams)], dtype=torch.short, device=device)

    for step_i in range(1, args.max_output_tokens + 1):

        # TODO: add preprocessing abstraction

        # transform each row of output_ids into tokens and print

        # A heap that will store all beam candidates, used to form the next beam.
        # This will include single-model advances (where one model is behind and is trying to catch up)
        # and paired-model advances (where both models advanced a compatible token).
        beam_candidates = []

        # A list of top-k items for each model from a synced state
        paired_topk = PairedTopk()

        # Take the next step of each model
        for model_i, bundle in enumerate(bundles):
            other_i = 1 - model_i
            step_outputs, next_token_scores = bundle.step(step_i)
            sequence_scores = next_token_scores + bundle.beam_scores[:, None].expand_as(next_token_scores)

            # do top-k selection where the model is behind or synced
            print("STEP", step_i, "MODEL", model_i, "SYNC", sync_states, (sync_states == 2) | (sync_states == model_i))
            bundle.print_beam(model_i, step_i)
            if any(region := ((sync_states == 2) | (sync_states == model_i))):
                next_indices, next_tokens, next_token_scores = bundle.topk(sequence_scores, region)
                # print("TOPK", model_i)
                # for i, cand in enumerate(bundle_topk[model_i]):
                #     # get the token str from the cand.token using the tokenizer
                #     token_str = bundle.tokenizer.convert_ids_to_tokens([cand.token])[0]
                #     print("->", i, cand, token_str)

                # create a candidate out of it
                for index, token, score in zip(next_indices[0], next_tokens[0], next_token_scores[0]):
                    print("CHECKING", model_i, index, token, score, sync_states[index])
                    # Advancing from a synced state is done by both models at once. This is handled below.
                    if sync_states[index] == 2:
                        cand = BeamItem(index, token, score, model_no=model_i)
                        paired_topk.add(model_i, cand)

                    elif sync_states[index] == model_i:
                        # Unsynced items can be dealt with immediately by comparing them to their corresponding
                        # beam item. If consistent, it can be added to the heap.
                        beam_str = bundles[other_i].get_hyp_str(index)
                        this_str = bundle.get_hyp_str(index, token)

                        if model_i == 0:
                            is_compat, new_sync_state = is_compatible(this_str, beam_str)
                            if is_compat:
                                pair = BeamItemPair(index, token, score, None, bundles[1].beam_scores[index], new_sync_state)
                                heapq.heappush(beam_candidates, pair)
                        elif model_i == 1:
                            is_compat, new_sync_state = is_compatible(beam_str, this_str)
                            if is_compat:
                                pair = BeamItemPair(index, None, bundles[0].beam_scores[index], token, score, new_sync_state)
                                heapq.heappush(beam_candidates, pair)

            # TODO: we may want to make two top-k calls per model, separating synced from non-synced rows for each model

            # token_str = "  ".join([f"{str(int(i))}/{bundle.tokenizer.decode(t, skip_special_tokens=False)}/{t}" for i, t in zip(next_indices[0, :], next_tokens[0, :])])
            # print("TOPK", model_i, token_str, next_token_scores)

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
        beam_selection = []  # contains selected items
        beam_completed = []  # contains completed sentences
        seen = set()  # popped items that shouldn't be appended again

        # Now add all compatible items to the queue
        # (efficiency will come later!)
        for index in paired_topk.get_beams():
            print("BEAM", index, "TOPK", paired_topk[index])
            for item0 in paired_topk[index][0]:
                for item1 in paired_topk[index][1]:
                    cand0_str = bundles[0].get_hyp_str(index, item0.token)
                    cand1_str = bundles[1].get_hyp_str(index, item1.token)
                    is_compat, new_sync_state = is_compatible(cand0_str, cand1_str)

                    heapq.heappush(beam_candidates, BeamItemPair(index, item0.token, item0.score, item1.token, item1.score, new_sync_state))

        print("GOT", len(beam_candidates), "CANDIDATES")
        beam_selection = beam_candidates[:num_beams]
            # print(step_i, "POP", pair.score(), pair.cand0_rank, pair.cand1_rank, cand0_str, "<>", cand1_str, is_compat)
            # if cand0.token == bundles[0].eos_token_id and cand1.token == bundles[1].eos_token_id:
            #     beam_completed.append((cand0, cand1))
            # else:
            #     beam_selection.append((cand0, cand1, new_sync_states))

        if len(beam_completed):
            print("DONEZO")
            break

        # Now enumerate the outputs and use them to update the beams in each sub-model
        if len(beam_selection) < num_beams:
            print("* FATAL: not enough candidates", len(beam_selection), "need", num_beams)
            sys.exit(1)

        beam_selection = beam_selection[:num_beams]
        for i in range(num_beams):
            cand0, cand1, new_sync_states = beam_selection[i]
            cand0_str = bundles[0].get_hyp_str(cand0.index, cand0.token)
            cand1_str = bundles[1].get_hyp_str(cand1.index, cand1.token)
            dir_str = "SAME" if new_sync_states == 2 else "DIFF"
            print("SELECTED", i, dir_str, cand0, cand0_str, cand1, cand1_str)
        
        # Now, go through and update each model with its selection.
        # There is a constraint on updating: a model can only update beam items
        # where the sync state is 2 (both models were already in sync) or where
        # the sync state is the model number (that model was behind).
        for i, bundle in enumerate(bundles):
            beam_tokens = [beam_selection[j][i].token for j in range(len(beam_selection))]
            beam_indices = [beam_selection[j][i].index for j in range(len(beam_selection))]
            beam_scores = [beam_selection[j][i].score for j in range(len(beam_selection))]
            filter = [sync_states[i] ]

            print("MODEL", i, "STEP", step_i, "UPDATE", beam_tokens, beam_indices)
            bundle.update(beam_tokens, beam_indices, beam_scores, sync_states)

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

def is_compatible(cand0_str, cand1_str):
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