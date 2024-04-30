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
        num_beams: Optional[int] = 5,
        max_length: Optional[int] = 256,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Adapted from `~transformers.generation_utils.GenerationMixin.beam_search` to accept a list of input_ids

        - Code source:
          https://github.com/huggingface/transformers/blob/07e3454f034b4889925621e8e3253547d2a04aa7/src/transformers/generation/utils.py#L2764
        - Beam search support:
          https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py
        """

        model.set_input(input, num_beams=num_beams, max_length=max_length)

        # if len(stopping_criteria) == 0:
        #     warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)

        # do an infinite loop with for over step_i
        step_i = 0
        while True:
            step_i += 1

            # TODO: add preprocessing abstraction
            step_outputs, next_token_scores = model.step(step_i)

            # Test adding random zeroes throughout
            # if cur_len % 2 == 0 and cur_len < 12:
            #     # append a column of zeros to output_ids
            #     print(f"[{cur_len}] Appending zeros")
            #     output_ids = torch.cat([output_ids, torch.zeros_like(output_ids[:, :1])], dim=-1)
            #     print(output_ids[0, :])

            ## 
            ## TODO (main): merge the outputs, create synced / unsynced beam item abstractions!
            ## 

            model.print_beam(step_i)

            sequence_scores = next_token_scores + model.beam_scores[:, None].expand_as(next_token_scores)

            next_indices, next_tokens, next_token_scores = model.topk(sequence_scores)

            beam_scores, beam_next_tokens, beam_idx = model.beam_select(next_token_scores, next_tokens, next_indices)

            model.update(beam_next_tokens, beam_idx, beam_scores, step_outputs)


            # test adding in random zeros, by replacing either the penultimate or last item in each
            # beam entry (dim 1) with a zero
            # if cur_len > 3 and cur_len < 10:
            #     output_ids = torch.cat([output_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            #     output_ids[:, -1] = tokenizer.pad_token_id
                # for i in range(len(output_ids)):
                #     # choose either -1 or -2
                #     index = random.choice([-1, -2])
                #     output_ids[i, index] = tokenizer.pad_token_id


            # model.print_beam(step)
            if model.is_done():
                break

        sequence_outputs = model.finalize()

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

        # normally you would now call beam search, but we need to implement it
        outputs = translate(line, model, num_beams=args.num_beams, max_length=args.max_output_tokens)

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
    parser.add_argument("--max-output-tokens", "-l", type=int, default=256, help="Maximum number of output tokens")
    args = parser.parse_args()

    main(args)