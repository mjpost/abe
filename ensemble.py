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
    def __init__(self, rank, index, token, score):
        self.rank = rank  # its rank in the model's topk list
        self.index = index  # the beam index it extends
        self.token = token  # the token itself
        self.score = score  # the score of the token

    def __lt__(self, other):
        return self.score < other.score


class BeamItemPair:
    """
    Represents a pair of beam items.
    """
    def __init__(self, cand1, cand2):
        self.cand1 = cand1
        self.cand2 = cand2

    def score(self):
        """
        Returns the interpolated score of the pair.
        """
        return (self.cand1.score + self.cand2.score) / 2

    def __lt__(self, other):
        return self.cand1.score() < other.cand1.score()


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
        output_scores: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict_in_generate: Optional[bool] = None,
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

    # instantiate beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        device=device,
    )

    for bundle in bundles:
        # Initialize each model with the input
        # TODO: maybe this is where the first decoder token should also be set?
        bundle.set_input(input, num_beams=num_beams)

    batch_size = 1  # len(beam_scorer._beam_hyps)
    batch_beam_size = batch_size * num_beams

    stopping_criteria = MaxLengthCriteria(max_length=max_length)

    # if len(stopping_criteria) == 0:
    #     warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    # pad_token_id = vocab.pad_token_id
    # eos_token_id = vocab.eos_token_id
    # if isinstance(eos_token_id, int):
    #     eos_token_id = [eos_token_id]
    # print("EOS", eos_token_id)

    # beam = EnsembleBeam()

    def compatible(cand1, cand2) -> Tuple[bool, int]:
        """
        """
        cand1_str = bundles[0].get_hyp_str(cand1.index, cand1.token)
        cand2_str = bundles[1].get_hyp_str(cand2.index, cand2.token)

        if len(cand1_str) < len(cand2_str):
            result = cand2_str.startswith(cand1_str), -1
        elif len(cand1_str) > len(cand2_str):
            result = cand1_str.startswith(cand2_str), 1
        else:
            result = cand1_str == cand2_str, 0

        print("CMP", cand1_str, " ||| ", cand2_str, "=", result)

        return result

    while (step := 1) < args.max_output_tokens:

        # TODO: add preprocessing abstraction

        # transform each row of output_ids into tokens and print
 
        candidates = [ [] for _ in range(num_models) ]

        # Take the next step of each model
        for model_i, bundle in enumerate(bundles):
            next_token_scores = bundle.step()

            bundle.print_beam(step)

                # if output_attentions:
                #     decoder_attentions += (
                #         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                #     )
                #     if self.config.is_encoder_decoder:
                #         cross_attentions += (outputs.cross_attentions,)

                # if output_hidden_states:
                #     decoder_hidden_states += (
                #         (outputs.decoder_hidden_states,)
                #         if self.config.is_encoder_decoder
                #         else (outputs.hidden_states,)
                #     )

            sequence_scores = next_token_scores + bundle.beam_scores[:, None].expand_as(next_token_scores)

            # topk on synchronized items
            next_indices, next_tokens, next_scores = bundle.topk(sequence_scores)  # , self.get_sync_mask())

            # hard-code one batch
            for rank, (index, token, score) in enumerate(zip(next_indices[0], next_tokens[0], next_scores[0])):
                # print("ITEM", index, token, score)
                cand = BeamItem(rank, index, token, score)
                candidates[model_i].append(cand)

            # topk on items where this model is behind
            # topk = bundle.topk(outputs, self.get_behind_mask(model_i))
            # for item in topk:
            #     # TODO: actually take the step, and push the item on the candidates list
            #     cand = Candidate(item)
            #     heapq.heappush(candidates[model_i], cand)

            for cand in candidates[model_i]:
                # concatenate the token to the beam
                print("CAND", bundle.get_hyp_str(cand.index, cand.token))

        # TODO: seed beam with the complete diagonal, and ensure each item is used only once
        q = [ BeamItemPair( candidates[0][0], candidates[1][0] ) ]
        selected = []
        while len(q) > 0 and len(selected) < len(candidates[0]) - 1:
            pair = heapq.heappop(q)
            cand1 = pair.cand1
            cand2 = pair.cand2
            if compatible(cand1, cand2):
                selected.append((cand1, cand2))
                # add successors
                heapq.heappush(q, BeamItemPair(candidates[0][cand1.rank + 1], candidates[1][cand1.rank + 1]))
            else:
                # add successors
                heapq.heappush(q, BeamItemPair(cand1, candidates[1][cand2.rank+1]))
                heapq.heappush(q, BeamItemPair(candidates[0][cand1.rank+1], cand2))

        for i, bundle in enumerate(bundles):
            next_token_scores = torch.tensor([selected[j][1].score for j in range(len(selected))], dtype=torch.float, device=device).unsqueeze(0)
            next_tokens = torch.tensor([selected[j][i].token for j in range(len(selected))], dtype=torch.long, device=device).unsqueeze(0)
            next_indices = torch.tensor([selected[j][i].index for j in range(len(selected))], dtype=torch.long, device=device).unsqueeze(0)
            # print("MODEL", i, next_token_scores, next_tokens, next_indices)
            beam_outputs = beam_scorer.process(
                bundle.output_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=bundle.pad_token_id,
                eos_token_id=bundle.eos_token_id,
                beam_indices=bundle.beam_indices,
                decoder_prompt_len=bundle.decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            bundle.output_ids = torch.cat([bundle.output_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

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
        scores = None
        if beam_scorer.is_done or stopping_criteria(bundle.output_ids, scores):
            print("MODEL", i, "IS DONE", beam_scorer.is_done, stopping_criteria(bundle.output_ids, scores))
            break

    sequence_outputs = []
    for i, bundle in enumerate(bundles):
        sequence_outputs.append(
            beam_scorer.finalize(
                bundle.output_ids,
                bundle.beam_scores,
                None,
                None,
                max_length=stopping_criteria.max_length,
                pad_token_id=bundle.pad_token_id,
                eos_token_id=bundle.eos_token_id,
                beam_indices=bundle.beam_indices,
                decoder_prompt_len=bundle.decoder_prompt_len,
            )
        )

    return sequence_outputs[0]["sequences"]


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
    parser.add_argument("--num-beams", "-b", type=int, default=2, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-output-tokens", "-l", type=int, default=30, help="Maximum number of output tokens")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    main(args)