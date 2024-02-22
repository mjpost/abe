#!/usr/bin/env python3

import argparse
import sys
import torch

from typing import Optional, Union, List, Dict, Any
from torch import nn, LongTensor

from transformers import (
    LogitsProcessor,
    BeamSearchScorer,
    BeamScorer,
    MaxLengthCriteria,
)

from transformers.generation.utils import (
    GenerateBeamOutput, 
    GenerateBeamDecoderOnlyOutput, 
    GenerateBeamEncoderDecoderOutput,
)
from models import get_model_bundle, Model
from vocab import SharedVocab

@torch.no_grad()
def ensemble_beam_search(
        input: str,
        models: List[Model],
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

        TODO:
        - [ ] Generalize to n models  
        """

        model = models[0]
        num_models = len(models)
        device = model.model.device

        num_beams = beam_scorer.num_beams

        for model in models:
            # Initialize each model with the input
            # TODO: maybe this is where the first decoder token should also be set?
            model.set_input(input, num_beams=num_beams)

        # This is used to store the canonical, shared output
        output_ids = torch.ones((num_beams, 1), device=model.model.device, dtype=torch.long)
        output_ids = output_ids * model.model.config.decoder_start_token_id

         # These are used to store the separate tokenizations from each model
        model_output_ids = [torch.ones((num_beams, 1), device=device, dtype=torch.long) for _ in models]
        for m in model_output_ids:
            m = m * model.model.config.decoder_start_token_id

        batch_size = len(beam_scorer._beam_hyps)
        batch_beam_size, cur_len = output_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )        

        stopping_criteria = MaxLengthCriteria(max_length=max_length)

        # Construct the shared model vocabulary. The beam works in this space.
        vocab = SharedVocab([model.get_vocab() for model in models])

        # if len(stopping_criteria) == 0:
        #     warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = vocab.pad_token_id
        eos_token_id = vocab.eos_token_id
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

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and config.is_encoder_decoder:
        #     encoder_attentions = model.model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model.model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=output_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        decoder_prompt_len = output_ids.shape[-1]  # record the prompt length of decoder
        while True:
            # TODO: add preprocessing abstraction

            # transform each row of output_ids into tokens and print
            # print("BEAM")
            # for i in range(output_ids.shape[0]):
            #     tokens = model.tokenizer.decode(output_ids[i].tolist())
            #     print("OUTPUT", beam_scores.view(batch_size, num_beams)[0][i], tokens, output_ids[i])
            # print()

            scores = []
            for modeli, model in enumerate(models):
                model_inputs = model.prepare_inputs_for_generation(model_output_ids[modeli])

                # Take the next step of the model
                outputs = model.step(model_inputs)

                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = model.logits_processor(output_ids, next_token_logits)
                next_token_scores = nn.functional.softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * num_beams, vocab_size)

                # next_token_scores_processed = model.logits_processor(output_ids, next_token_scores)
                scores.append(next_token_scores)

            """
            TODO: merge scores from different vocabularies
            Each entry in `scores` is a tensor of shape (batch_size * num_beams, vocab_size).
            We need to project each, using the `SharedVocab` object, into the shared vocabulary space.
            This gives the ID in each original vocabulary that can be used to directly interpolate.
            We then fast-forward multi-token models to catch up.
            Not sure yet how to handle the fact that there will be different output lengths for each model.
            """

            ## 1. Project scores from each model into the shared vocabulary space and interpolate
            projected_scores = [vocab.project_scores(s, i) for i, s in enumerate(scores)]
            projected_scores = torch.stack(scores, dim=0)
            projected_scores = torch.mean(projected_scores, dim=0)
            next_token_scores = projected_scores
            next_token_scores = torch.log(next_token_scores)        
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )

            ## 2. Make the selection for each beam item


            ## 3. Advance all models so that the beam is synchronized across tokenizations




            # log softmax the scores

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

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

            # extend the sequence of generated output tokens
            # The output IDs are in the shared vocabulary space; each model will map them back to their own vocabulary space
            output_ids = torch.cat([output_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # I don't know what this is so I'm commenting it out
            # if model_kwargs["past_key_values"] is not None:
            #     warnings.warn(f"past_key_values are not supported for ensemble generation")
            #     model_kwargs["past_key_values"] = self._temporary_reorder_cache(
            #         model_kwargs["past_key_values"], beam_idx
            #     )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

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

    models = []
    for model_name in args.model_names:
        models.append(get_model_bundle(model_name, target_language=args.target_lang))

    if args.noise is not None:
        models[0].logits_processor.append(
            RandomNoiseLogitsProcessor(args.noise)
        )

    for line in sys.stdin:
        line = line.rstrip()

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=args.num_beams,
            device=models[0].model.device,
        )

        # normally you would now call beam search, but we need to implement it
        outputs = ensemble_beam_search(line, models, beam_scorer, max_length=args.max_output_tokens)

        # decode with the combined vocabulary
        result = models[0].tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", "-m", type=str, nargs="+", default=["facebook/m2m100_418M", "facebook/m2m100_418M"], help="Model names")
    parser.add_argument("--target-lang", "-t", type=str, default="fra_Latn", help="Target language")
    parser.add_argument("--num-beams", "-b", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-output-tokens", "-l", type=int, default=30, help="Maximum number of output tokens")
    args = parser.parse_args()

    main(args)