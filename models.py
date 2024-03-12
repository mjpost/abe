import copy
import sys
import torch

from typing import Optional, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer

from transformers import (
    LogitsProcessorList,
    ForcedBOSTokenLogitsProcessor,
    StoppingCriteriaList,
)

from transformers.generation.utils import (
    ModelOutput,
)

def get_model_bundle(
        model_name: str,
        target_language: Optional[str] = "fr",
        ) -> "Bundle":

    print(f"Instantiating model {model_name}", file=sys.stderr)

    if model_name == "facebook/nllb-200-distilled-600M":
        lang_map = {
            "fr": "fra_Latn",
        }

        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
        tokenizer = NllbTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        bos_token_id = tokenizer.lang_code_to_id[lang_map[target_language]]
        print("*", model_name, bos_token_id, file=sys.stderr)
        return Bundle(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id, is_encoder_decoder=True)

    elif model_name == "facebook/m2m100_418M":
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang="en", tgt_lang=target_language)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_token_id = tokenizer.lang_code_to_id[target_language]
        print("*", model_name, bos_token_id, file=sys.stderr)
        return Bundle(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id, is_encoder_decoder=True)

    elif model_name == "facebook/m2m100_1.2B":
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang="en", tgt_lang=target_language)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_token_id = tokenizer.lang_code_to_id[target_language]
        print("*", model_name, bos_token_id, file=sys.stderr)
        return Bundle(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id, is_encoder_decoder=True)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


class ModelState:
    def __init__(self,
                 bundle: "Bundle",
    ):
        self.model = bundle
        self.model_kwargs = bundle.model_kwargs
        self.model_input = bundle.input
        self.model_output = None
        self.model_output_ids = None
        self.model_beam_scores = None
        self.model_beam_indices = None


class Bundle:
    def __init__(self,
                 model: Optional[PreTrainedModel] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 max_length: Optional[int] = None,
                 bos_force_token: Optional[int] = None,
                 is_encoder_decoder: Optional[bool] = False,
                 k=1,
                 device=None,
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_force_token = bos_force_token
        self.model_kwargs = kwargs
        self.k = k
        self.device = device

        # TODO: use config.is_encoder_decoder
        self.is_encoder_decoder = is_encoder_decoder

        # These are used as a cache in step(), drastically speeds things up
        self.output_attentions = False
        self.output_hidden_states = False

        self.encoder_attentions = None
        self.encoder_hidden_states = None
        self.encoder_outputs = None

        self.logits_processor = LogitsProcessorList()
        if self.bos_force_token is not None:
            self.logits_processor.append(
                ForcedBOSTokenLogitsProcessor(bos_force_token)
        )

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def set_input(self, line: str, num_beams, return_tensors="pt"):
        """
        Tokenize and encode the input. For seq2seq models, store the encoder outputs
        for passing in the generation loop in step().
        """
        input = self.tokenizer(line, return_tensors=return_tensors)
        encoder_input_ids = input.input_ids

        # Encoder outputs only need to be set once, then stored
        self.encoder_outputs = self.model.get_encoder()(
            encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        )

        self.encoder_attentions = None
        self.encoder_hidden_states = None
        if self.config.is_encoder_decoder:
            self.encoder_attentions = self.encoder_outputs.get("attentions")
            self.encoder_hidden_states = (
                self.encoder_outputs.get("hidden_states")
            )

        self.model_kwargs = {
            "encoder_outputs": self.encoder_outputs,
        }

        # TODO: figure out what generation_config is
        generation_config = copy.deepcopy(self.model.generation_config)
        self.model_kwargs = generation_config.update(**self.model_kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(self.model_kwargs.copy())

        # Now: initialize the output states
        model_output_ids = torch.ones((num_beams, 1), device=self.device, dtype=torch.long)
        model_output_ids = model_output_ids * self.model.config.decoder_start_token_id

        if self.model.bos_force_token:
            model_inputs = self.model.prepare_inputs_for_generation(model_output_ids)
            model_inputs = self.prepare_inputs_for_generation(model_output_ids, **self.model_kwargs)

            # Step
            step_outputs = self.model.step(model_inputs)
            next_token_logits = step_outputs.logits[:, -1, :]
            # print("* OUTPUTS.LOGITS", next_token_logits.shape)

            forced_tokens = torch.ones((num_beams, 1), device=self.device, dtype=torch.long) * self.model.bos_force_token
            model_output_ids = torch.cat([model_output_ids, forced_tokens], dim=-1)

            # Initialize models, including running over force BOS tokens
        # These store individual models' tokenized outputs
        self.model_output_ids.append(model_output_ids)

        return encoder_input_ids, encoder_outputs
    
    def prepare_inputs_for_generation(self, outputs):
        """
        This is called at every step. It provides a way to modify the inputs before they are passed to the model.
        """
        return self.model.prepare_inputs_for_generation(outputs, **self.model_kwargs)

    def step(self, model_inputs):
        """
        Takes a step in the generation loop.
        """
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )
        return outputs

    def topk(self, model_inputs, mask, k=None):
        k = self.k if k is None else k

        # Give the model its current outputs
        # print("MODEL", modeli, "GIVING INPUTS", model_output_ids)
        model_inputs = self.prepare_inputs_for_generation(model_inputs)

        # Step
        step_outputs = self.step(model_inputs)
        next_token_logits = step_outputs.logits[:, -1, :]
        # print("* OUTPUTS.LOGITS", next_token_logits.shape)

        # Massage the logits. This is how prefix decoding is enforced.
        next_token_logits = self.logits_processor(model_output_ids, next_token_logits)
        next_token_scores = nn.functional.softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        # next_token_scores_processed = model.logits_processor(output_ids, next_token_scores)
        # print(STEP, "Max score from model", modeli, torch.max(next_token_scores, dim=-1))
        scores.append(next_token_scores)

        next_token_scores = next_token_scores + model_beam_scores[:, None].expand_as(next_token_scores)

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
        )

        # print("TOPK", next_tokens.shape, next_tokens)

        # TODO: create candidate items out of these
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        return next_indices, next_tokens, next_token_scores


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