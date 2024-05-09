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

    # def __lt__(self, other):
    #     return self.score < other.score


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


# class PairedTopk:
#     """
#     Represents a set of top-k items for each model.
#     """
#     def __init__(self):
#         self.items = {}

#     def add(self, model_i, beam_index, token, score):
#         """
#         Append an item to the top-k list.
#         """
#         beam_i = int(beam_index)
#         if beam_i not in self.items:
#             self.items[beam_i] = [[], []]  # a list of extensions for this beam item for both models

#         self.items[beam_i][model_i].append((beam_index, token, score))

#     def __str__(self):
#         if self.model_no:
#             return f"ITEM({self.index}, {self.token}, {self.score} #{self.model_no})"
#         else:
#             return f"ITEM({self.index}, {self.token}, {self.score})"

#     def get_beams(self):
#         """
#         Return the list of beam indices.
#         """
#         return [key for key in self.items.keys() if len(self.items[key][0]) > 0 and len(self.items[key][1]) > 0]

#     def __getitem__(self, index):
#         return self.items[index]

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
        """
        """
        cand1_str = bundles[0].get_surface_str(cand1.index, cand1.token)
        cand2_str = bundles[1].get_surface_str(cand2.index, cand2.token)

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

    beam_completed = []  # contains completed sentences

    """
    # Models: 0 (TM) and 1 (LLM)
    BEAM 0: [Life is like a box of], [Life is like a box of]    # -> sync state 2
    BEAM 1: [Life is like a container], [Life is like a cont]   # -> sync state 1 (model 1 is shorter)
    BEAM 2: [Life is similar to a], [Life is similar to a box]  # -> sync state 0 (model 0 is shorter)
    """
    for step_i in range(1, args.max_output_tokens + 1):

        # TODO: add preprocessing abstraction

        # transform each row of output_ids into tokens and print

        # A heap that will store all beam candidates, used to form the next beam.
        # This will include single-model advances (where one model is behind and is trying to catch up)
        # and paired-model advances (where both models advanced a compatible token).
        beam_candidates = []

        paired_topk = defaultdict(lambda: defaultdict(list))

        # Take the next step of each model
        for model_i, bundle in enumerate(bundles):
            other_i = 1 - model_i
            step_outputs, next_token_scores = bundle.step(step_i)
            sequence_scores = next_token_scores + bundle.beam_scores[:, None].expand_as(next_token_scores)

            # do top-k selection where the model is behind or synced
            print("STEP", step_i, "MODEL", model_i, "SYNC", sync_states)
            bundle.print_beam(model_i, step_i)
            if any(region := ((sync_states == 2) | (sync_states == model_i))):
                next_indices, next_tokens, next_token_scores = bundle.topk(sequence_scores, region)
                # print("TOPK", model_i)
                # for i, cand in enumerate(bundle_topk[model_i]):
                #     # get the token str from the cand.token using the tokenizer
                #     token_str = bundle.tokenizer.convert_ids_to_tokens([cand.token])[0]
                #     print("->", i, cand, token_str)

                # create a candidate out of it
                for beam_index, token, score in zip(next_indices[0], next_tokens[0], next_token_scores[0]):
                    beam_index = int(beam_index)

                    # print("ITEM model", model_i, "beam", int(beam_index), token, score, bundle.id_to_token(token))
                    # Advancing from a synced state is done by both models at once. This is handled below.
                    if sync_states[beam_index] == 2:
                        paired_topk[beam_index][model_i].append(BeamItem(model_i, beam_index, token, score))

                    elif sync_states[beam_index] == model_i and token != bundle.eos_token_ids:
                        # Unsynced items can be dealt with immediately by comparing them to their corresponding
                        # beam item. If consistent, it can be added to the heap.
                        beam_str = bundles[other_i].get_surface_str(beam_index)  # e.g., "Life is like a box"
                        this_str = bundle.get_surface_str(beam_index, token)     # e.g., "Life is like a" + "box" -> "Life is like a box" -> 2

                        # Unfortunately we need to handle the models separately to get the arguments right
                        if model_i == 0:
                            is_compat, new_sync_state = is_compatible(this_str, beam_str)
                            if is_compat:
                                pair = BeamItemPair(
                                    BeamItem(model_i, beam_index, token, score), 
                                    BeamItem(other_i, beam_index, bundles[1].pad_token_id, bundles[1].beam_scores[beam_index]), 
                                    new_sync_state)
                                heapq.heappush(beam_candidates, pair)
                        elif model_i == 1:
                            is_compat, new_sync_state = is_compatible(beam_str, this_str)
                            if is_compat:
                                pair = BeamItemPair(
                                    BeamItem(other_i, beam_index, bundles[0].pad_token_id, bundles[0].beam_scores[beam_index]), 
                                    BeamItem(model_i, beam_index, token, score), 
                                    new_sync_state)
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

        """
        paired_topk: top level is beam index (0, 1,, ..., k)
        Within each, is two lists: extensions of that beam item in model 0, and model 1
        Think of this as a 2d grid, what we'll do is explore the *entire* grid (not worrying about
        efficiency for now) to find the pairs in the grid that are compatible, i.e., that generate compatible
        strings.

        prefix: Life is like a (sync state 2)

        model0: box, cont, shipping_crate, cardboard_box
        model1: box, container, shipp, holder

        compat: (box, box) -> 2, (cont, container) -> 0, (shipping_crate, shipp) -> 1
        """

        # Now handle paired items, e.g., beam steps taken from synced states
        for beam_index in sorted(paired_topk.keys()):
            # print("BEAM", beam_index, "ITEMS", len(paired_topk[beam_index][0]), "x", len(paired_topk[beam_index][1]))
            used_j = set()
            for i, item0 in enumerate(  paired_topk[beam_index][0]):
                for j, item1 in enumerate(paired_topk[beam_index][1]):
                    if j in used_j:
                        continue

                    # for synced states, both hyps have to finish together
                    if bundles[0].is_eos(item0.token) != bundles[1].is_eos(item1.token):
                        continue

                    # check if the pair is compatible and if so, add it to the beam_candidates queue
                    cand0_str = bundles[0].get_surface_str(beam_index, item0.token)
                    cand1_str = bundles[1].get_surface_str(beam_index, item1.token)
                    is_compat, new_sync_state = is_compatible(cand0_str, cand1_str)

                    if is_compat:
                        used_j.add(j)
                        heapq.heappush(beam_candidates, BeamItemPair(
                            BeamItem(0, beam_index, item0.token, item0.score), 
                            BeamItem(1, beam_index, item1.token, item1.score), 
                            new_sync_state)
                        )
                        break

        # TODO: there might be no overlap in topk
        # solution: repeat, with 2x k
        # another solution: take each model's topk, individually look those ids up in the other model
        #   however: this requires a context-sensitive vocab overlap (recall that the vocabs are distinct)
        # Doesn't seem likely that the models would have zero overlap.
        # This might be more likely if the LLM is used only as a language model (and doesn't see the source)
        # But this can be determined empirically, could be an interesting study.
        # Could get tricky in LM-only mode with items that can't be predicted from context, e.g., novel name mentions.
        # Alternative: if no overlaps, let one of the models make the next selection. Probably the TM.
        # It will then be ahead everywhere, and the LLM will have to catch up. The advantage is no new call to topk() 
        # is needed (which might fail again, e.g., for a really out-of-domain name).

        # TODO: seed beam with the complete diagonal, and ensure each item is used only once

        # Take the top k as the next beam
        print("GOT", len(beam_candidates), "CANDIDATES")

        # TODO: make sure we have enough candidates to fill the beam

        # The next beam
        beam_selection = []
        while len(beam_selection) < num_beams and len(beam_candidates):
            cand = heapq.heappop(beam_candidates)
            print("STEP", step_i, "CAND", cand, cand.cand0.token, bundles[0].eos_token_ids, cand.cand1.token, bundles[1].eos_token_ids)
            if bundles[0].is_eos(cand.cand0.token) and bundles[1].is_eos(cand.cand1.token):
                print(step_i, "COMPLETE", cand)
                beam_completed.append((bundles[0].output_ids[cand.cand0.index], bundles[0].beam_scores[cand.cand0.index], 
                                       bundles[1].output_ids[cand.cand1.index], bundles[1].beam_scores[cand.cand1.index], cand))
            else:
                beam_selection.append(cand)

        """
        TODO:
        - TM is going to generate </s> and EOS, LLM needs to ignore
          - We use [eos] as a separator, big LLMs will not have this
          - Big LLMs: when do they generated </s>? Just at end-of-response, right? I don't think they mark internal sentences.
            bundles[0].delete_from_string_tokens = ["[eos]"]
          - Who controls when the end comes (probably the TM)
          - bundle.step() needs to handle beam items with different lengths

        In normal beam search, every beam item has the exact same length. This is not the case here.

        normal beam: everything the same length
        next step, computes self attention (len 4)

            BEAM 0: a b c d
            BEAM 1: b f g e
            BEAM 2: a x j d

        What are the inputs to producing the distribution over the vocab for the next step?

        bundle 0:
            BEAM 0: a b c d _
            BEAM 1: b f g e a
            BEAM 2: a x j _ _ 
        """


            # print(step_i, "POP", pair.score(), pair.cand0_rank, pair.cand1_rank, cand0_str, "<>", cand1_str, is_compat)
            # if cand0.token == bundles[0].eos_token_id and cand1.token == bundles[1].eos_token_id:
            #     beam_completed.append((cand0, cand1))
            # else:
            #     beam_selection.append((cand0, cand1, new_sync_states))

        # Now enumerate the outputs and use them to update the beams in each sub-model
        if len(beam_selection) < num_beams:
            print("* WARNING: not enough candidates", len(beam_selection), "need", num_beams)

            if len(beam_completed) > 0:
                break

        # PICKUP figuring out when you're done
        print("I COUNT", len(beam_completed), "COMPLETED")
        if len(beam_completed) == num_beams:
            print("DONE")
            break

        for i in range(num_beams):
            cand0, cand1, new_sync_states = beam_selection[i].cand0, beam_selection[i].cand1, beam_selection[i].sync_state
            cand0_str = bundles[0].get_surface_str(cand0.index, cand0.token)
            cand1_str = bundles[1].get_surface_str(cand1.index, cand1.token)
            print("SELECTED", i, beam_selection[i], cand0_str, "|||", cand1_str)
        
        # Now, go through and update each model with its selection.
        # There is a constraint on updating: a model can only update beam items
        # where the sync state is 2 (both models were already in sync) or where
        # the sync state is the model number (that model was behind).
        for i, bundle in enumerate(bundles):
            update_mask = [sync_states[j] == 2 or sync_states[j] == i for j in range(len(sync_states))]
            beam_indices = [beam_selection[j][i].index for j in range(len(beam_selection))]
            beam_tokens = [beam_selection[j][i].token if update_mask[j] else bundle.pad_token_id for j in range(len(beam_selection))]
            beam_scores = [beam_selection[j][i].score for j in range(len(beam_selection))]

            print("MODEL", i, "STEP", step_i, "UPDATE", beam_indices, beam_tokens, update_mask)
            bundle.update(beam_indices, beam_tokens, beam_scores)

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

        # update sync states
        sync_states = torch.tensor([beam_selection[j].sync_state for j in range(len(beam_selection))], dtype=torch.short, device=device)

    for i, completed in enumerate(beam_completed):
        model0_ids, model0_score, model1_ids, model1_score, pair = completed
        model0_str = bundles[0].tokenizer.decode(model0_ids, skip_special_tokens=True)
        print(f"COMPLETED {i}:", model0_str, model0_score, model1_score, pair)

    return model0_str, pair.score()

    # for bundle in bundles:
    #     sequence_outputs = bundle.finalize()
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

    # return sequence_outputs["sequences"]

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

    # print("-> IS_COMPAT", cand0_str, "<>", cand1_str, is_compat, new_sync_state)

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
        print(outputs[0], outputs[1])

        # decode with the combined vocabulary
        # result = models[0].tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print(result)


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