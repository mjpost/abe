#!/usr/bin/env python3

"""
I'm using this script just to test translation, completely unwrapped from the Hugging Face library.
"""


import copy
import sys
import torch

from typing import Any, Dict, Optional, Union, List
from torch import nn, LongTensor

from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    BeamSearchScorer,
    BeamScorer,
    PreTrainedModel, 
    PreTrainedTokenizer,
    ForcedBOSTokenLogitsProcessor,
    StoppingCriteriaList,
)

from transformers.generation.utils import (
    GenerateBeamOutput, 
    GenerateBeamDecoderOnlyOutput, 
    GenerateBeamEncoderDecoderOutput,
    ModelOutput,
)
from transformers.generation.stopping_criteria import MaxLengthCriteria

from models import get_model_bundle, Bundle


@torch.no_grad()
def translate(
        input: str,
        model: Bundle,
        beam_scorer: BeamScorer,
        max_length: Optional[int] = None,
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
        """

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        model.set_input(input, num_beams=num_beams)

        # init values
        if max_length is not None:
            # warnings.warn(
            #     "`max_length` is deprecated in this function, use"
            #     " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            #     UserWarning,
            # )
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        tokenizer = model.tokenizer
        vocab = model.tokenizer.get_vocab()

        # if len(stopping_criteria) == 0:
        #     warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores
        output_attentions = ( output_attentions )
        output_hidden_states = ( output_hidden_states )
        return_dict_in_generate = ( return_dict_in_generate )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        config = model.model.config

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # These are used only when returning the results at the end, not here for computation or caching.
        if return_dict_in_generate and config.is_encoder_decoder:
            encoder_attentions = model.model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model.model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=model.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        for step in range(1, args.max_output_tokens + 1):
            # TODO: add preprocessing abstraction
            step_outputs, next_token_scores = model.step()

            # Test adding random zeroes throughout
            # if cur_len % 2 == 0 and cur_len < 12:
            #     # append a column of zeros to output_ids
            #     print(f"[{cur_len}] Appending zeros")
            #     output_ids = torch.cat([output_ids, torch.zeros_like(output_ids[:, :1])], dim=-1)
            #     print(output_ids[0, :])

            ## 
            ## TODO (main): merge the outputs, create synced / unsynced beam item abstractions!
            ## 

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (step_outputs.decoder_attentions,) if config.is_encoder_decoder else (step_outputs.attentions,)
                    )
                    if config.is_encoder_decoder:
                        cross_attentions += (step_outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (step_outputs.decoder_hidden_states,)
                        if config.is_encoder_decoder
                        else (step_outputs.hidden_states,)
                    )

            sequence_scores = next_token_scores + model.beam_scores[:, None].expand_as(next_token_scores)

            next_indices, next_tokens, next_token_scores = model.topk(sequence_scores)

            # stateless
            beam_outputs = beam_scorer.process(
                model.output_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=model.decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # extend the sequence of generated output tokens
            model.output_ids = torch.cat([model.output_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # test adding in random zeros, by replacing either the penultimate or last item in each
            # beam entry (dim 1) with a zero
            # if cur_len > 3 and cur_len < 10:
            #     output_ids = torch.cat([output_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            #     output_ids[:, -1] = tokenizer.pad_token_id
                # for i in range(len(output_ids)):
                #     # choose either -1 or -2
                #     index = random.choice([-1, -2])
                #     output_ids[i, index] = tokenizer.pad_token_id

            model.update_kwargs(step_outputs, beam_idx)

            # model.print_beam(step)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            if beam_scorer.is_done or stopping_criteria(model.output_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            model.output_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=model.decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if model.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]


class RandomNoiseLogitsProcessor(LogitsProcessor):
    def __init__(self, noise):
        self.noise = noise

    def __call__(self, 
                 input_ids: LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
            return scores + torch.randn_like(scores) * self.noise


def main(args):

    model = get_model_bundle(args.model_name, target_language=args.target_lang)

    if args.noise is not None:
        model.logits_processor.append(
            RandomNoiseLogitsProcessor(args.noise)
        )

    for line in sys.stdin:
        line = line.rstrip()

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=args.num_beams,
            device=model.model.device,
        )

        # normally you would now call beam search, but we need to implement it
        outputs = translate(line, model, beam_scorer, max_length=args.max_output_tokens)

        # decode with the combined vocabulary
        result = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, default="facebook/m2m100_418M", help="Model name")
    parser.add_argument("--target-lang", "-t", type=str, default="fra_Latn", help="Target language")
    parser.add_argument("--num-beams", "-b", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-output-tokens", "-l", type=int, default=100, help="Maximum number of output tokens")
    args = parser.parse_args()

    main(args)