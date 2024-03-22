#!/usr/bin/env python3

import copy
import sys
import torch
import warnings

from typing import Optional, Union, List
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
from transformers.generation.stopping_criteria import validate_stopping_criteria


def get_model_bundle(
        model_name: str,
        target_language: Optional[str] = None,
        ) -> "Model":
    if model_name == "facebook/nllb-200-distilled-600M":
        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
        tokenizer = NllbTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        bos_token_id = tokenizer.lang_code_to_id["fra_Latn"]

        return Model(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id)
    elif model_name == "facebook/m2m100_418M":
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_token_id = tokenizer.lang_code_to_id[target_language]

        return Model(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id, is_encoder_decoder=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class Model:
    def __init__(self,
                 model: Optional[PreTrainedModel] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
                 stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
                 max_length: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[Union[int, List[int]]] = None,
                 bos_force_token: Optional[int] = None,
                 is_encoder_decoder: Optional[bool] = False,
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_force_token = bos_force_token
        self.model_kwargs = kwargs

        # TODO: use config.is_encoder_decoder
        self.is_encoder_decoder = is_encoder_decoder

        self.output_attentions = False
        self.output_hidden_states = False

        if self.bos_force_token is not None:
            self.logits_processor.append(
                ForcedBOSTokenLogitsProcessor(bos_force_token)
        )
            
        self.input = None

    def set_input(self, line: str, num_beams, return_tensors="pt"):
        self.input = self.tokenizer(line, return_tensors=return_tensors)
        encoder_input_ids = self.input.input_ids

        self.model_kwargs = {
            "encoder_outputs": self.model.get_encoder()(
                encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
            ),
        }
        generation_config = copy.deepcopy(self.model.generation_config)
        self.model_kwargs = generation_config.update(**self.model_kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(self.model_kwargs.copy())

        return encoder_input_ids
    
    def prepare_inputs_for_generation(self, inputs):
        return self.model.prepare_inputs_for_generation(inputs, **self.model_kwargs)

    def step(self, model_inputs):
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )
        return outputs


    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        # Bloom fix: standardizes the cache format when requested
        # if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
        #     batch_size = outputs.logits.shape[0]
        #     past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values


    def update_model_kwargs_for_generation(self, outputs: ModelOutput):
        # update past_key_values
        self.model_kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)
        if getattr(outputs, "state", None) is not None:
            self.model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in self.model_kwargs:
            token_type_ids = self.model_kwargs["token_type_ids"]
            self.model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not self.is_encoder_decoder:
            # update attention mask
            if "attention_mask" in self.model_kwargs:
                attention_mask = self.model_kwargs["attention_mask"]
                self.model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in self.model_kwargs:
                decoder_attention_mask = self.model_kwargs["decoder_attention_mask"]
                self.model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

@torch.no_grad()
def ensemble_beam_search(
        input: str,
        model: Model,
        beam_scorer: BeamScorer,
        max_length: Optional[int] = None,
        output_scores: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
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

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        model.set_input(input, num_beams=num_beams)
        output_ids = torch.ones((num_beams, 1), device=model.model.device, dtype=torch.long)
        output_ids = output_ids * model.model.config.decoder_start_token_id

        batch_beam_size, cur_len = output_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )        

        # init values
        logits_processor = model.logits_processor
        stopping_criteria = model.stopping_criteria
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

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
        if return_dict_in_generate and config.is_encoder_decoder:
            encoder_attentions = model.model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model.model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=output_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only

        decoder_prompt_len = output_ids.shape[-1]  # record the prompt length of decoder
        while True:
            # TODO: do this separately for each model
            # TODO: add preprocessing abstraction
            # TODO: why is this done at every step? shouldn't it be done just once, outside the loop?
            model_inputs = model.prepare_inputs_for_generation(output_ids)

            # TODO: once per model
            outputs = model.step(model_inputs)

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # TODO: once per model
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits_processed = logits_processor(output_ids, next_token_logits)

            next_token_scores = nn.functional.log_softmax(
                next_token_logits_processed, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )

            ## 
            ## TODO (main): merge the outputs, create synced / unsynced beam item abstractions!
            ## 

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

            # stateless
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
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

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
        outputs = ensemble_beam_search(line, model, beam_scorer, max_length=args.max_output_tokens)

        # decode with the combined vocabulary
        result = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, default="facebook/nllb-200-distilled-600M", help="Model name")
    parser.add_argument("--target-lang", "-t", type=str, default="fra_Latn", help="Target language")
    parser.add_argument("--num-beams", "-b", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-output-tokens", "-l", type=int, default=30, help="Maximum number of output tokens")
    args = parser.parse_args()

    main(args)