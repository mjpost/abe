import copy
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
        target_language: Optional[str] = None,
        ) -> "Model":
    
    target_language = target_language or "fra_Latn"
    if model_name == "facebook/nllb-200-distilled-600M":
        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
        tokenizer = NllbTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        bos_token_id = tokenizer.lang_code_to_id[target_language]
        return Model(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id)

    elif model_name == "facebook/m2m100_418M":
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_token_id = tokenizer.lang_code_to_id[target_language]
        return Model(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id, is_encoder_decoder=True)

    elif model_name == "facebook/m2m100_1.2B":
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