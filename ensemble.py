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


STEP = 0

@torch.no_grad()
def ensemble_beam_search(
        input: str,
        models: List[Model],
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

        # This is used to store the canonical, shared output
        output_ids = torch.ones((num_beams, 1), device=model.model.device, dtype=torch.long)
        output_ids = output_ids * model.model.config.decoder_start_token_id

         # These are used to store the separate tokenizations from each model
        model_output_ids = [torch.ones((num_beams, 1), device=device, dtype=torch.long) for _ in models]
        for i in range(len(model_output_ids)):
            model_output_ids[i] = model_output_ids[i] * model.model.config.decoder_start_token_id

        print("MODEL_OUTPUT_IDS", model_output_ids)

        batch_size = 1  # len(beam_scorer._beam_hyps)
        batch_beam_size, cur_len = output_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        stopping_criteria = MaxLengthCriteria(max_length=max_length)

        # Construct the shared model vocabulary. The beam works in this space.
        vocab = SharedVocab([model.tokenizer for model in models])

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
        max_steps = 5
        while True:
            STEP = output_ids[0].shape[-1]
            if len(output_ids[0]) > max_steps:
                print("Breaking after", max_steps, "steps")
                break

            # TODO: add preprocessing abstraction

            # transform each row of output_ids into tokens and print
            print("BEAM")
            for i in range(output_ids.shape[0]):
                tokens = vocab.decode(output_ids[i].tolist())
                # print(i, output_ids[i].tolist())
                print(i, beam_scores.view(batch_size, num_beams)[0][i], tokens, output_ids[i])
            print()

            # Take the next step of each model
            scores = []
            for modeli, model in enumerate(models):
                # Give the model its current outputs
                print("MODEL", modeli, "GIVING INPUTS", model_output_ids[modeli])
                model_inputs = model.prepare_inputs_for_generation(model_output_ids[modeli])

                # Step
                step_outputs = model.step(model_inputs)
                next_token_logits = step_outputs.logits[:, -1, :]
                # print("* OUTPUTS.LOGITS", next_token_logits.shape)
                next_token_logits = model.logits_processor(model_output_ids[modeli], next_token_logits)
                next_token_scores = nn.functional.softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * num_beams, vocab_size)

                # next_token_scores_processed = model.logits_processor(output_ids, next_token_scores)
                print(STEP, "Max score from model", modeli, torch.max(next_token_scores, dim=-1))
                scores.append(torch.tensor(vocab.project_into(modeli, next_token_scores, step=STEP), device=device))

            """
            TODO: merge scores from different vocabularies
            Each entry in `scores` is a tensor of shape (batch_size * num_beams, vocab_size).
            We need to project each, using the `SharedVocab` object, into the shared vocabulary space.
            This gives the ID in each original vocabulary that can be used to directly interpolate.
            We then fast-forward multi-token models to catch up.
            Not sure yet how to handle the fact that there will be different output lengths for each model.
            """

            ## Project scores from each model into the shared vocabulary space for interpolation
            # print(stepno, "Max score from model 0", torch.max(scores[0]))
            # print(stepno, "Max score from model 1", torch.max(scores[1]))
            projected_scores = torch.stack(scores, dim=0)
            next_token_scores = torch.mean(projected_scores, dim=0)
            # print(stepno, "Max score after mean", torch.max(next_token_scores, dim=-1))
            # print(stepno, "Min score after mean", torch.min(next_token_scores, dim=-1))
            next_token_scores = torch.log(next_token_scores)
            # print(stepno, "Max score after interpolation", torch.max(next_token_scores))
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            print("TOPK", next_tokens.shape, next_tokens)

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


                # for beam_i in range(num_beams):

                #     for next_token in new_token_seq[beam_i, 1:]:

                #     for token in new_token_seq[beam_i]:
                #         if token == 0:
                #             break

                #         # add the token to the outputs
                #         print("MODEL", modeli, "OLD BEAM", model_output_ids[modeli][beam_i, :])
                #         print("NEW TOKEN", token)
                #         model_output_ids[modeli][beam_i, :] = torch.cat([model_output_ids[modeli][beam_i, :], torch.tensor([token], device=device)], dim=-1)

                #         # call prepare
                #         model_inputs = model.prepare_inputs_for_generation(model_output_ids[modeli])

                #         # step, force-decoding to the token
                #         outputs = model.step(model_inputs, force_token_id=next_token)

                #     # project the token back into the model's space
                #     token_id = vocab.project_token(token_id, modeli)
                #     # add the token to the model's output
                #     model_output_ids[modeli][beam_i, cur_len] = token_id


                # # for modeli, model in enumerate(models):
                #     # Give the model its current outputs
                #     # model_inputs = model.prepare_inputs_for_generation(model_output_ids[modeli])

                #     # # Step
                #     # outputs = model.step(model_inputs)

                #     # next_token_logits = outputs.logits[:, -1, :]
                #     # next_token_logits = model.logits_processor(output_ids, next_token_logits)
                #     # next_token_scores = nn.functional.softmax(
                #     #     next_token_logits, dim=-1
                #     # )  # (batch_size * num_beams, vocab_size)

            # Temporary fix for when beam size == 1
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
    args = parser.parse_args()

    main(args)