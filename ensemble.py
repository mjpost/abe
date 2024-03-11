#!/usr/bin/env python3

import argparse
import heapq
import sys
import torch

from typing import Optional, Union, List, Dict, Any
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
from vocab import SharedVocab

__version__ = "0.0.1"

STEP = 0

class BeamItem:
    def __init__(self, score, tokens):
        self.score = score
        self.tokens = [x for x in tokens]
        self.synced = True


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


class Candidate:
    def __init__(self, item):
        pass


@torch.no_grad()
def ensemble_beam_search(
        input: str,
        models: List[Bundle],
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

    num_models = len(models)
    device = models[0].model.device

    # instantiate beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        device=device,
    )

    for model in models:
        # Initialize each model with the input
        # TODO: maybe this is where the first decoder token should also be set?
        model.set_input(input, num_beams=num_beams)

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

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = [torch.zeros((batch_size, num_beams), dtype=torch.float, device=device) for _ in range(num_models)]
    for i in range(num_models):
        beam_scores[i][:, 1:] = -1e9
        beam_scores[i] = beam_scores[i].view((batch_size * num_beams,))

    beam = EnsembleBeam()

    model_output_ids = []

    def compatible(cand1, cand2):
        return models[0].tokenizer.decode(cand1.tokens) == models[0].tokenizer.decode(cand2.tokens)

    STEP = 0
    while True:
        STEP += 1

        # TODO: add preprocessing abstraction

        # transform each row of output_ids into tokens and print
        # print_beam(num_beams, vocab, output_ids, batch_size, beam_scores, STEP)
 
        candidates = [ [] for _ in range(num_models) ]

        # Take the next step of each model
        for model_i, model in enumerate(models):
            model_inputs = model.prepare_inputs_for_generation(model_output_ids[model_i])
            outputs = model.step(model_inputs)

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


            # topk on synchronized items
            topk = model.topk(outputs, self.get_sync_mask())
            for item in topk:
                cand = Candidate(item)
                heapq.heappush(candidates[model_i], cand)

            # topk on items where this model is behind
            topk = model.topk(outputs, self.get_behind_mask(model_i))
            for item in topk:
                # TODO: actually take the step, and push the item on the candidates list
                cand = Candidate(item)
                heapq.heappush(candidates[model_i], cand)

        lazy_items = []
        for i, cand1 in enumerate(candidates[0]):
            for j, cand2 in enumerate(candidates[1]):
                if compatible(cand1, cand2):
                    heapq.heappush(lazy_items, LazyItem(cand1, cand2))
                    candidates[0].remove(cand1)
                    candidates[1].remove(cand2)
        for j, cand2 in enumerate(candidates[1]):
            for i, cand1 in enumerate(candidates[0]):
                if compatible(cand1, cand2):
                    heapq.heappush(lazy_items, LazyItem(cand1, cand2))
                    candidates[0].remove(cand1)
                    candidates[1].remove(cand2)

        # Now, we do lazy k-best extraction from the pairs
        next_beam = EnsembleBeam()
        while len(next_beam) < num_beams:
            # pop top item from lazy_items
            pair = heapq.heappop(lazy_items)

            # add to next_beam
            next_beam.append(pair)

            # add sucessors
            for succ in pair.successors():
                heapq.heappush(lazy_items, succ)

        beam = next_beam

        """
        TODO: merge scores from different vocabularies
        Each entry in `scores` is a tensor of shape (batch_size * num_beams, vocab_size).
        We need to project each, using the `SharedVocab` object, into the shared vocabulary space.
        This gives the ID in each original vocabulary that can be used to directly interpolate.
        We then fast-forward multi-token models to catch up.
        Not sure yet how to handle the fact that there will be different output lengths for each model.
        """

        # Why does the beam scorer care about the decoder prompt length?
        beam_outputs = beam_scorer.process(
            output_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        print("-> NEXT TOKENS", beam_next_tokens.shape, beam_next_tokens, vocab.decode(beam_next_tokens.tolist()))
        print("-> NEXT INDICES", beam_idx)

        # extend the sequence of generated output tokens
        # The output IDs are in the shared vocabulary space; each model will map them back to their own vocabulary space
        print("TOP BEAM", output_ids[beam_idx, :])
        print("NEXT TOKENS", beam_next_tokens.unsqueeze(-1))
        output_ids = torch.cat([output_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        # MJP: I don't know what this is so I'm commenting it out
        # if model_kwargs["past_key_values"] is not None:
        #     warnings.warn(f"past_key_values are not supported for ensemble generation")
        #     model_kwargs["past_key_values"] = self._temporary_reorder_cache(
        #         model_kwargs["past_key_values"], beam_idx
        #     )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        # We now have a new token for every beam, but this token is in the global vocabulary.
        # For each model, we need to convert the set of tokens into the model's private vocabulary
        # space. Unfortunately, these tokenizations may be of different lengths. We find the maximum
        # tokenization length and pad the others.
        for modeli, model in enumerate(models):
            # For each beam, tokenize the token using the private vocabulary
            new_token_seq = [torch.tensor(vocab.project_outta(token_id, modeli), device=device) for token_id in beam_next_tokens]
            # Turn that into a ragged zero-padded tensor
            new_token_seq = torch.nn.utils.rnn.pad_sequence(new_token_seq, batch_first=True, padding_value=0).unsqueeze(-1)

            # Add the first new token column to the model's output
            print("CONCAT MODEL", modeli, "OLD BEAM", model_output_ids[modeli])
            print("  CONCATTING NEW_TOKEN_SEQ", new_token_seq[:, 0])
            model_output_ids[modeli] = torch.cat([model_output_ids[modeli], new_token_seq[:, 0]], dim=-1)

            """
            Consider a beam size of 3, and assume we have the following tokens in the global vocab: [ 3, 192, 99 ].
            This gets tokenized into each model's vocab, which tokenization may be of different lengths. e.g.,

            [ [ 17, 52, 0 ],   # 3 in global vocab
                [ 9,  0,  0 ],   # 192 in global vocab
                [ 14, 12, 0 ] ]  # 99 in global vocab

            We now proceed column by column. We first concat the first column onto the beam. If there are more columns
            (which there are here), we step the decoder, row by row, since we don't want to take a step on a zero.
            """

            # For each column of new tokens
            for next_tokeni in range(1, new_token_seq.shape[1]):

                # Go beam by beam, since we have to test for zeroes
                for beam_i in range(num_beams):
                    next_token = new_token_seq[beam_i, next_tokeni]
                    if next_token == 0:
                        continue

                    # take a step in the model with the token
                    model_inputs = model.prepare_inputs_for_generation(model_output_ids[modeli])
                    # Update the outputs
                    step_outputs = model.step(model_inputs)
                    # next_token_logits = outputs.logits[:, -1, :]
                    # next_token_logits = model.logits_processor(output_ids, next_token_logits)
                    # next_token_scores = nn.functional.softmax(
                    #     next_token_logits, dim=-1
                    # )

                # for each beam, add the current column to the outputs
                model_output_ids[modeli] = torch.cat([model_output_ids[modeli], new_token_seq[:, next_tokeni]], dim=1)

                # TODO: shift left over previous zeros anywhere

        if beam_scorer.is_done or stopping_criteria(output_ids, scores):
            break

    sequence_outputs = beam_scorer.finalize(
        output_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    return sequence_outputs["sequences"]


def print_beam(num_beams, vocab, output_ids, batch_size, beam_scores, STEP):
    print("BEAM", STEP)
    for i in range(output_ids.shape[0]):
        tokens = vocab.decode(output_ids[i].tolist())
            # print(i, output_ids[i].tolist())
        print(i, beam_scores.view(batch_size, num_beams)[0][i], tokens, output_ids[i])
    print()


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