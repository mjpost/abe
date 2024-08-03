import copy
import sys
import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Any, Dict, Optional, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer, NoBadWordsLogitsProcessor

from transformers import (
    LogitsProcessorList,
    ForcedBOSTokenLogitsProcessor,
    BeamSearchScorer,
    StoppingCriteriaList,
)

from transformers.generation.utils import (
    ModelOutput,
)

from transformers.generation.stopping_criteria import MaxLengthCriteria


def get_model_bundle(
        model_name: str,
        target_language: Optional[str] = "fr",
        source_language: Optional[str] = "en",
        device: Optional[torch.device] = None,
        cache: Optional[bool] = False,
        ) -> "Bundle":

    print(f"* Instantiating model {model_name}", file=sys.stderr)

    if model_name == "facebook/nllb-200-distilled-600M":
        # eventual todo: move into a json/different file to import/load etc
        lang_map = {
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "en": "eng_Latn",
        }

        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
        tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang=lang_map[source_language], tgt_lang=lang_map[target_language])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        bos_force_token = tokenizer.convert_tokens_to_ids([lang_map.get(target_language, target_language)])[0]

    elif model_name in ["facebook/m2m100_418M", "facebook/m2m100_1.2B"]:
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang=source_language, tgt_lang=target_language)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_force_token = tokenizer.convert_tokens_to_ids([f"__{target_language}__"])[0]

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.eval()
    model = model.to(device)

    return Bundle(model=model, tokenizer=tokenizer, bos_force_token=bos_force_token, is_encoder_decoder=True, device=device, cache=cache)



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
                 bos_force_token: Optional[int] = None,
                 is_encoder_decoder: Optional[bool] = False,
                 device=None,
                 model_i: Optional[int] = None,
                 cache: Optional[bool] = False,
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.model_kwargs = kwargs
        self.cache = cache
        self.device = device
        self.model_i = model_i

        self.num_beams = None
        self.batch_size = None
        self.batch_beam_size = None

        # use this to record historical beam indices if you choose
        self.beam_indices = None

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_ids = self.tokenizer.eos_token_id
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]
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

        self.bad_words = [_ for _ in tokenizer.added_tokens_decoder.keys() if _ not in [tokenizer.eos_token_id]]
        
        self.logits_processor = LogitsProcessorList(
        )
        if self.bos_force_token is not None:
            self.logits_processor.append(
                ForcedBOSTokenLogitsProcessor(bos_force_token)
        )

        # init values
        self.stopping_criteria = None

        self.decoder_prompt_len = None

        # instantiate beam scorer
        self.beam_scorer = None

        self.legacy_added_tokens = set(self.tokenizer._added_tokens_encoder.keys()) - set(self.tokenizer.all_special_tokens) | {
            token for token in self.tokenizer.additional_special_tokens if self.tokenizer.convert_tokens_to_ids(token) >= self.tokenizer.vocab_size
        }
        self.skip_special_vocab_itos = [_[0] if (_[1] not in self.bad_words and not self.is_eos(_[1])) else "" for _ in sorted(self.tokenizer.get_vocab().items(), key=lambda kv: kv[1])]
        self.vocab_itos = [_[0] for _ in sorted(self.tokenizer.get_vocab().items(), key=lambda kv: kv[1])]

        self.speed_vocab = True
        self.vocab_size = len(self.skip_special_vocab_itos)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def set_input(self, line: str, num_beams, max_length=256, return_tensors="pt"):
        """
        Tokenize and encode the input. For seq2seq models, store the encoder outputs
        for passing in the generation loop in step().
        """
        input = self.tokenizer(line, return_tensors=return_tensors)
        encoder_input_ids = input.input_ids.to(self.device)

        self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        # instantiate beam scorer
        self.beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beams,
            device=self.device,
        )

        self.batch_size = 1
        self.num_beams = num_beams
        self.batch_beam_size = self.batch_size * num_beams

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
            "use_cache": self.cache
        }

        # TODO: figure out what generation_config is
        generation_config = copy.deepcopy(self.model.generation_config)
        self.model_kwargs = generation_config.update(**self.model_kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(self.model_kwargs.copy())

        # Now: initialize the output states
        self.output_ids = torch.ones((num_beams, 1), device=self.device, dtype=torch.long) * self.model.config.decoder_start_token_id
        self.decoder_tokens = [[self.model.config.decoder_start_token_id] for _ in range(num_beams)]
        self.decoder_prefixes = [[] for _ in range(num_beams)]


        batch_size = 1
        batch_beam_size, cur_len = self.output_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )        


        # Initialize models, including running over force BOS tokens
        if self.bos_force_token:
            forced_tokens = torch.ones((num_beams, 1), device=self.device, dtype=torch.long) * self.bos_force_token
            self.output_ids = torch.cat([self.output_ids, forced_tokens], dim=-1)
            self.decoder_tokens = [self.decoder_tokens[i] + [self.bos_force_token] for i in range(num_beams)]
            # self.step(sequential=False)

        self.decoder_prompt_len = self.output_ids.shape[-1]

        return encoder_input_ids, self.encoder_outputs
    
    def step(self, step_no=None, sequential=False):
        """
        Takes a step in the generation loop after preparing the inputs.
        """      

        # 0 is masked, 1 is not masked
        step_inputs = self.prepare_inputs_for_generation(self.output_ids, **self.model_kwargs)
        attention_mask = (self.output_ids != self.pad_token_id).int().to(self.device)
        step_outputs = self.model(
            **step_inputs,
            decoder_attention_mask=attention_mask,
            return_dict=True,
        )

        # Get the logits for the next token
        # torch selection for BATCH X SEQUENCE LENGTH X VOCABULARY
        # torch.arange will create a index for each batch index position
        # then we select the last non pad item in each sequence
        # last item is length of sequence -1 to account for 0-indexing
        # next_token_logits = step_outputs.logits[torch.arange(4), ((self.output_ids != self.pad_token_id).sum(dim=1)-1)]
        next_token_logits = step_outputs.logits[torch.arange(4), ((step_inputs['decoder_input_ids'] != self.pad_token_id).sum(dim=1)-1)]
        # next_token_logits = step_outputs.logits[:, -1, :]


        # Massage the logits. This is how prefix decoding is enforced.
        next_token_logits_processed = self.logits_processor(self.output_ids, next_token_logits)
        next_token_logits_processed[:, self.bad_words] = -float("inf")
        next_token_scores = torch.nn.functional.log_softmax(
            next_token_logits_processed, dim=-1
        )  # (batch_size * num_beams, vocab_size)


        return step_outputs, next_token_scores
    

    def truncate_past_key_values(self, past_key_values, min_length):
        """
        Truncate the past key values to num_beams.
        """
        new_past_key_values = []
        for layer in past_key_values:
            new_past_key_values.append([])
            for tup in layer:
                new_past_key_values[-1].append(tup[:, :, :min_length, :])
        return tuple([tuple(layer) for layer in new_past_key_values])


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def extend_beam_string(self, beam_index, token_id):
        """
        Extends the beam string with the specified token.
        """
        sequence = self.decoder_prefixes[beam_index] + [token_id]
        return self._decode(sequence, clean_up_tokenization_spaces=False, skip_special_tokens=True)



    def get_surface_str(self, beam_index, token_id=None):
        """
        Returns the surface string of the specified beam index, optionally with the specified token appended.
        """
        sequence = self.output_ids[beam_index]
        if token_id is not None:
            sequence = torch.cat([sequence, torch.tensor([token_id], device=self.device)], dim=-1)
        return self._decode(sequence, clean_up_tokenization_spaces=False, skip_special_tokens=True)

    def id_to_token(self, token_id, skip_special_tokens=False):
        search_in = self.vocab_itos if not skip_special_tokens else self.skip_special_vocab_itos
        if self.speed_vocab:
            if self.vocab_size > token_id:
                return search_in[token_id]
            else:
                return "<unk>" if not skip_special_tokens else ""
        else:
            return self.tokenizer.convert_ids_to_tokens([token_id])[0]


    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        # from this file https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/tokenization_utils.py#L405
        tokenizer = self.tokenizer
        tokenizer._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # Convert the token IDs to strings
        filtered_tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id, skip_special_tokens=skip_special_tokens)
            filtered_tokens.append(token)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        # TODO @ArthurZ in version 5, special tokens should be handled in convert_tokens_to_string, while _convert_tokens_to_string
        for token in filtered_tokens:
            if token in self.legacy_added_tokens:
                if current_sub_text:
                    string = tokenizer.convert_tokens_to_string(current_sub_text, skipped=skip_special_tokens)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text, skipped=skip_special_tokens))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else tokenizer.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = tokenizer.clean_up_tokenization(text)
            return clean_text
        else:
            return text
        
    def convert_tokens_to_string(self, tokens, skipped=True):
        # from this file https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/tokenization_utils.py#L405
        """Converts a sequence of tokens (string) in a single string."""
        tokenizer = self.tokenizer
        
        current_sub_tokens = []
        out_string = ""
        
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if skipped or token in tokenizer.all_special_tokens:
                out_string += tokenizer.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += tokenizer.sp_model.decode(current_sub_tokens)
        return out_string

    def print_beam(self, model_i=None, step=None):
        if model_i is not None:
            print(f"BEAM {step} MODEL {model_i}")
        else:
            print("BEAM", step)
        for i in range(self.output_ids.shape[0]):
            tokens = self.tokenizer.convert_ids_to_tokens(self.output_ids[i].tolist())
            token_str = self.tokenizer.decode(self.output_ids[i], skip_special_tokens=True)
            print(i, f"len={len(tokens)}", self.beam_scores.view(self.batch_size, self.num_beams)[0][i], " ".join(tokens), self.output_ids[i], token_str)
        print()

    def topk(self, sequence_scores, region=None, multiplier=3):
        """
        Select the top k items, ignoring any masked items.
        If set, `region` is used to limit the top-k selection to the specified rows of the beam.
        """
        k = self.num_beams

        # Give the model its current outputs
        # print("MODEL", modeli, "GIVING INPUTS", model_output_ids)

        # set all values to -inf in rows not in the selectable region
        if region is not None and not torch.all(region):
            sequence_scores = sequence_scores.clone().masked_fill((~region).unsqueeze(-1), float("-inf"))

        # reshape for beam search
        vocab_size = sequence_scores.shape[-1]
        sequence_scores = sequence_scores.view(self.batch_size, k * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        # MJP: In the worst case, we'd select all EOS tokens. To avoid that, we ensure that k is set such
        # that we have at least 1 non-EOS token per beam.
        n_eos_tokens = len(self.eos_token_ids) if self.eos_token_ids else 0
        num_wanted = k * max(multiplier, 1 + n_eos_tokens)
        sequence_scores, next_tokens = torch.topk(
            sequence_scores, num_wanted, dim=1, largest=True, sorted=True
        )

        # TODO: create candidate items out of these
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        token_str = " ".join([f"{i}/{t}/{self.id_to_token(t)}" for i, t in zip(next_indices[0].tolist(), next_tokens[0].tolist())])
        print("TOPK", next_tokens.shape, token_str)

        return next_indices, next_tokens, sequence_scores

    def beam_select(self,
                    next_token_scores,
                    next_tokens,
                    next_indices):
        """
        Uses the beam scorer to select the tokens that can be used for the next beam.
        """
        # stateless
        beam_outputs = self.beam_scorer.process(
            self.output_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_ids,
            beam_indices=self.beam_indices,
            decoder_prompt_len=self.decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        return beam_scores, beam_next_tokens, beam_idx

    def update(self, beam_idx, beam_next_tokens, beam_scores, step_outputs=None, debug=False):
        """
        Updates the model's beam sequences by extended the specified beam indices with the selected tokens.
        If {step_outputs} is defined, it will update the cache, as well.

        :param beam_next_tokens: The next tokens to append to the beam sequences
        :param beam_idx: The beam indices that are being extended
        :param beam_scores: The beam scores of each item, to add to the running totals
        :param step_outputs: The model outputs from the current step
        """
        # if type(beam_next_tokens) is not torch.Tensor:
        #     beam_next_tokens = torch.tensor(beam_next_tokens, device=self.device, dtype=torch.long)
        # if type(beam_idx) is not torch.Tensor:
        #     beam_idx = torch.tensor(beam_idx, device=self.device, dtype=torch.long)
        if type(beam_scores) is not torch.Tensor:
            beam_scores = torch.tensor(beam_scores, device=self.device, dtype=torch.float)

        # update the beam scores
        self.beam_scores = beam_scores

        if debug:
            print("UPDATE", beam_idx, beam_next_tokens, beam_scores, file=sys.stderr)

        # extend the sequence of generated output tokens
        next_beam_tokens = []
        next_decoder_prefixes = []
        for i in range(self.num_beams):
            if beam_next_tokens[i] != self.tokenizer.pad_token_id:    
                if self.resets_prefix(i, beam_next_tokens[i]):
                    next_decoder_prefixes.append([beam_next_tokens[i]])
                else:
                    next_decoder_prefixes.append(self.decoder_prefixes[beam_idx[i]] + [beam_next_tokens[i]])
                next_beam_tokens.append(self.decoder_tokens[beam_idx[i]] + [beam_next_tokens[i]])
            else:
                next_decoder_prefixes.append(self.decoder_prefixes[beam_idx[i]])
                next_beam_tokens.append(self.decoder_tokens[beam_idx[i]])
        self.decoder_tokens = next_beam_tokens
        self.decoder_prefixes = next_decoder_prefixes
        self.output_ids = pad_sequence([torch.tensor(_) for _ in self.decoder_tokens], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)

        if step_outputs is not None:
            step_outputs['past_key_values'] = self.truncate_past_key_values(step_outputs['past_key_values'], torch.min((self.output_ids != self.pad_token_id).sum(dim=1)-1))
            self.model_kwargs = self._update_model_kwargs_for_generation(
                step_outputs, self.model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if self.model_kwargs["past_key_values"] is not None:
                self.model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    self.model_kwargs["past_key_values"], torch.tensor(beam_idx, device=self.device)
                )

    def resets_prefix(self, beam_i, addition):
        """
        Returns true if the prefix resets the sequence.
        """
        surface = self.extend_beam_string(beam_i, addition)
        if len(surface.split()) > 1 or surface[-1] == " ":
            return True
        return False

    def is_eos(self, token):
        return int(token) in self.eos_token_ids

    def beam_is_done(self):
        """
        Returns true if the beam is complete.
        """
        beam_scorer_done = self.beam_scorer.is_done
        stopping_criteria_done = self.stopping_criteria(self.output_ids, self.beam_scores)
        print("IS_DONE", "beam=", beam_scorer_done, "stop=", stopping_criteria_done)
        return beam_scorer_done or stopping_criteria_done

    def beam_item_is_done(self, beam_index):
        """
        Returns true if the last item in a beam index is the EOS token.
        Function is pad_token_id aware.
        """
        return self.output_ids[beam_index][self.output_ids[beam_index] != self.pad_token_id][-1] in self.eos_token_ids

    def finalize(self):
        return self.beam_scorer.finalize(
            self.output_ids,
            self.beam_scores,
            None,
            None,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_ids,
            max_length=self.stopping_criteria.max_length,
            beam_indices=self.beam_indices,
            decoder_prompt_len=self.decoder_prompt_len,
        )

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

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.

        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
        model_class = self.model.__class__.__name__.lower()
        # Exception 1: code path for models using the legacy cache format
        if isinstance(past_key_values, (tuple, list)):
            past_key_values = self.model._reorder_cache(past_key_values, beam_idx)
        # Exception 2: models with different cache formats. These are limited to `DynamicCache` until their
        # cache format is standardized, to avoid adding complexity to the codebase.
        elif "bloom" in model_class or "gptbigcode" in model_class:
            if not isinstance(past_key_values, DynamicCache):
                raise ValueError(
                    f"Using an unsupported cache format with {model_class}. Currently, it only supports the "
                    "legacy tuple format or `DynamicCache`"
                )
            past_key_values = self.model._reorder_cache(past_key_values, beam_idx)
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # Standard code path: use the `Cache.reorder_cache`
        else:
            past_key_values.reorder_cache(beam_idx)
        return past_key_values


def _split(data, full_batch_size: int, split_size: int = None):
    """
    Takes care of three cases:
    1. data is a tensor: e.g. last_hidden_state, pooler_output etc. split them on the batch_size dim
    2. data is a tuple: e.g. hidden_states, attentions etc. Keep the tuple as it is and split each tensor in it and
       return a list of tuples
    3. data is a tuple of tuples, e.g. past_key_values. Keep the tuple as it is and split each tuple in it and
       return a list of tuples of tuples
    (see documentation of ModelOutput)
    """
    if data is None:
        return [None] * (full_batch_size // split_size)
    if isinstance(data, torch.Tensor):
        return [data[i : i + split_size] for i in range(0, full_batch_size, split_size)]
    elif isinstance(data, tuple):
        # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
        if isinstance(data[0], tuple):
            return [
                tuple(tuple(tensor[i : i + split_size] for tensor in inner_tuple) for inner_tuple in data)
                for i in range(0, full_batch_size, split_size)
            ]

        else:
            return [
                tuple(sub_tensor[i : i + split_size] for sub_tensor in data)
                for i in range(0, full_batch_size, split_size)
            ]
    else:
        raise ValueError(f"Unexpected attribute type: {type(data)}")


def _split_model_inputs(
    model_input: Union[ModelOutput, Dict], split_size: int, full_batch_size: int
) -> List[Union[ModelOutput, Dict]]:
    """
    Split a ModelOutput object (or its subclasses) or Dict into a list of same-class objects based on a specified split
    size. The input object is dict when it was prepared for forward pass and ModelOutput when it was returned from
    previous forward pass.
    """
    # Edge case: if model_input is None, return a list of Nones
    # this happens with Whisper where encoder_outputs is None
    if model_input is None:
        return [model_input] * (full_batch_size // split_size)
    # Infer the class from the object
    model_output_cls = type(model_input)
    if (full_batch_size % split_size) != 0:
        raise ValueError("`full_batch_size` must be divisible by `split_size`")

    if split_size > full_batch_size:
        raise ValueError("`split_size` must be smaller or equal to `full_batch_size`")

    # Helper function to split tensors or tuples of tensors

    # Find all the dataclass fields (e.g., last_hidden_state, pooler_output etc.) and split them
    keys = (
        model_input.__dataclass_fields__.keys() if hasattr(model_input, "__dataclass_fields__") else model_input.keys()
    )
    # We only keep keys that are in the model_input
    keys = [k for k in keys if k in model_input]
    # Here we can have four types of values: tensors, tuples of tensors and booleans, and encoder_outputs which is a
    # ModelOutput object.
    # bool should not be split but replicated for each split
    bool_keys = [k for k in keys if isinstance(model_input[k], bool) or k == "cache_position"]
    keys_to_ignore = ["cache_position", "encoder_outputs"]
    non_bool_keys = [k for k in keys if not isinstance(model_input[k], bool) and k not in keys_to_ignore]

    # we split the tensors and tuples of tensors
    data_split_list = [
        {k: _split(model_input[k], full_batch_size, split_size)[i] for k in non_bool_keys}
        for i in range(full_batch_size // split_size)
    ]
    # bool values are the same and replicated for each split
    bool_data = {k: model_input[k] for k in bool_keys}
    # encoder_outputs is a ModelOutput object and should be split by its own
    if "encoder_outputs" in model_input:
        encoder_outputs_split = _split_model_inputs(model_input["encoder_outputs"], split_size, full_batch_size)
        data_split_list = [
            {**data_split, "encoder_outputs": encoder_outputs_split[i]} for i, data_split in enumerate(data_split_list)
        ]

    # Convert each dictionary in the list to an object of the inferred class
    split_model_inputs: List[Union[ModelOutput, Dict]] = [
        model_output_cls(**data_split, **bool_data) for data_split in data_split_list
    ]

    return split_model_inputs


def stack_model_outputs(model_outputs: List[ModelOutput]) -> ModelOutput:
    """
    Stack a list of ModelOutput objects (or its subclasses) along the batch_size dimension. The function infers the
    specific ModelOutput subclass from the list provided.
    """
    if not model_outputs:
        raise ValueError("Input list is empty.")

    # Infer the class from the first object in the list
    model_output_cls = type(model_outputs[0])

    # Ensure all objects are of the same type
    if not all(isinstance(obj, model_output_cls) for obj in model_outputs):
        raise ValueError("All elements in the list should be of the same type.")

    # Helper function to concat tensors or tuples of tensors
    def _concat(data):
        """
        Reverse of `_split` function above.
        """
        if any(data is None for data in data):
            return None
        if isinstance(data[0], torch.Tensor):
            return torch.cat(data, dim=0)
        elif isinstance(data[0], tuple):
            # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
            if isinstance(data[0][0], tuple):
                return tuple(
                    tuple(torch.cat([attr[i][j] for attr in data], dim=0) for j in range(len(data[0][0])))
                    for i in range(len(data[0]))
                )
            else:
                return tuple(torch.cat([attr[i] for attr in data], dim=0) for i in range(len(data[0])))
        elif isinstance(data[0], (int, float)):
            # If the elements are integers or floats, return a tensor
            return torch.tensor(data)
        else:
            raise ValueError(f"Unexpected attribute type: {type(data[0])}")

    # Use a dictionary comprehension to gather attributes from all objects and concatenate them
    concatenated_data = {
        k: _concat([getattr(model_output, k) for model_output in model_outputs])
        for k in model_output_cls.__dataclass_fields__.keys()
    }

    # Return a new object of the inferred class with the concatenated attributes
    return model_output_cls(**concatenated_data)
