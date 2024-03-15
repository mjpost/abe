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
        bos_force_token = tokenizer.lang_code_to_id[lang_map[target_language]]
        print("*", model_name, bos_force_token, file=sys.stderr)
        return Bundle(model=model, tokenizer=tokenizer, bos_force_token=bos_force_token, is_encoder_decoder=True)

    elif model_name == "facebook/m2m100_418M":
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang="en", tgt_lang=target_language)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_force_token = tokenizer.lang_code_to_id[target_language]
        print("*", model_name, bos_force_token, file=sys.stderr)
        return Bundle(model=model, tokenizer=tokenizer, bos_force_token=bos_force_token, is_encoder_decoder=True)

    elif model_name == "facebook/m2m100_1.2B":
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang="en", tgt_lang=target_language)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_force_token = tokenizer.lang_code_to_id[target_language]
        print("*", model_name, bos_force_token, file=sys.stderr)
        return Bundle(model=model, tokenizer=tokenizer, bos_force_token=bos_force_token, is_encoder_decoder=True)

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
                 batch_size=1,
                 device=None,
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_kwargs = kwargs
        self.k = k
        self.device = device
        self.batch_size = batch_size

        batch_beam_size = batch_size * k
        # use this to record historical beam indices if you choose
        self.beam_indices = None
        # self.beam_indices = (
        #     tuple(() for _ in range(batch_beam_size))
        # )

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]
        self.bos_force_token = bos_force_token

        # TODO: use config.is_encoder_decoder
        self.is_encoder_decoder = is_encoder_decoder

        # These are used as a cache in step(), drastically speeds things up
        self.output_attentions = False
        self.output_hidden_states = False

        self.encoder_attentions = None
        self.encoder_hidden_states = None
        self.encoder_outputs = None

        # used to hold the generated tokens
        self.output_ids = None

        self.beam_scores = None

        self.logits_processor = LogitsProcessorList()
        if self.bos_force_token is not None:
            self.logits_processor.append(
                ForcedBOSTokenLogitsProcessor(bos_force_token)
        )
            
        self.decoder_prompt_len = None

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def set_input(self, line: str, num_beams, return_tensors="pt"):
        """
        Tokenize and encode the input. For seq2seq models, store the encoder outputs
        for passing in the generation loop in step().
        """
        input = self.tokenizer(line, return_tensors=return_tensors)
        encoder_input_ids = input.input_ids

        self.k = num_beams

        # Encoder outputs only need to be set once, then stored
        self.encoder_outputs = self.model.get_encoder()(
            encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        )

        self.beam_scores = torch.zeros((self.batch_size, num_beams), dtype=torch.float, device=self.device)
        self.beam_scores[:, 1:] = -1e9
        self.beam_scores = self.beam_scores.view((self.batch_size * num_beams,))

        self.encoder_attentions = None
        self.encoder_hidden_states = None
        if self.is_encoder_decoder:
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
        self.output_ids = torch.ones((num_beams, 1), device=self.device, dtype=torch.long)
        self.output_ids = self.output_ids * self.model.config.decoder_start_token_id
        print("MODEL START TOKEN", self.model.config.decoder_start_token_id)

        # Initialize models, including running over force BOS tokens
        if self.bos_force_token:

            forced_tokens = torch.ones((num_beams, 1), device=self.device, dtype=torch.long) * self.bos_force_token
            self.output_ids = torch.cat([self.output_ids, forced_tokens], dim=-1)

            self.step()
            # step_outputs = self.model.step()
            # next_token_logits = step_outputs.logits[:, -1, :]
            # print("* OUTPUTS.LOGITS", next_token_logits.shape)

        # These store individual models' tokenized outputs
        # self.model_output_ids.append(self.output_ids)

        self.decoder_prompt_len = self.output_ids.shape[-1]

        return encoder_input_ids, self.encoder_outputs
    
    def step(self):
        """
        Takes a step in the generation loop after preparing the inputs.
        """

        step_inputs = self.model.prepare_inputs_for_generation(self.output_ids, **self.model_kwargs)

        step_outputs = self.model(
            **step_inputs,
            return_dict=True,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        next_token_logits = step_outputs.logits[:, -1, :]
        # print("* OUTPUTS.LOGITS", next_token_logits.shape)

        # Massage the logits. This is how prefix decoding is enforced.
        next_token_logits_processed = self.logits_processor(self.output_ids, next_token_logits)
        next_token_scores = torch.nn.functional.softmax(
            next_token_logits_processed, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        return next_token_scores

    def get_hyp_str(self, beam_index, token_id):
        sequence = torch.cat([self.output_ids[beam_index], torch.tensor([token_id])], dim=-1)
        return self.tokenizer.decode(sequence, skip_special_tokens=True)

    def print_beam(self, step=None):
        print("BEAM", step)
        for i in range(self.output_ids.shape[0]):
            tokens = self.tokenizer.decode(self.output_ids[i].tolist())
                # print(i, output_ids[i].tolist())
            print(i, self.beam_scores.view(self.batch_size, self.k)[0][i], tokens, self.output_ids[i])
        print()

    def topk(self, scores, mask=None, k=None):
        k = self.k if k is None else k

        # Give the model its current outputs
        # print("MODEL", modeli, "GIVING INPUTS", model_output_ids)

        # reshape for beam search
        vocab_size = scores.shape[-1]
        scores = scores.view(self.batch_size, k * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        # MJP: I don't understand the logic here. For some reason, we want _at least_ 2x as many topk
        #      candidates as there are beams.
        n_eos_tokens = len(self.eos_token_id) if self.eos_token_id else 0
        scores, next_tokens = torch.topk(
            scores, max(2, 1 + n_eos_tokens) * k, dim=1, largest=True, sorted=True
        )

        # print("TOPK", next_tokens.shape, next_tokens)

        # TODO: create candidate items out of these
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        return next_indices, next_tokens, scores

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

    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.

        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
        model_class = self.__class__.__name__.lower()
        # Exception 1: code path for models using the legacy cache format
        if isinstance(past_key_values, (tuple, list)):
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
        # Exception 2: models with different cache formats. These are limited to `DynamicCache` until their
        # cache format is standardized, to avoid adding complexity to the codebase.
        elif "bloom" in model_class or "gptbigcode" in model_class:
            if not isinstance(past_key_values, DynamicCache):
                raise ValueError(
                    f"Using an unsupported cache format with {model_class}. Currently, it only supports the "
                    "legacy tuple format or `DynamicCache`"
                )
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # Standard code path: use the `Cache.reorder_cache`
        else:
            past_key_values.reorder_cache(beam_idx)
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