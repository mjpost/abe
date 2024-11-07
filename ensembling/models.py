import os, sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")


from typing import Any, Dict, Optional, Union, List
import copy

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM

from transformers.generation.utils import (
    ModelOutput,
)

import torch
from torch.nn.utils.rnn import pad_sequence

from utils import tokenize

def get_models(model_names, device, cache):
    models = []

    for model_i, model_name in enumerate(model_names):
        print(f"Instantiating model {model_name}", file=sys.stderr)
        try:
            config = AutoConfig.from_pretrained(model_name)
            source_tokenizer = AutoTokenizer.from_pretrained(model_name)
            target_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if config.is_encoder_decoder:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error instantiating model {model_name}: {e}")
            sys.exit(-1)

        model.eval().to(device)

        models.append(Model(source_tokenizer, target_tokenizer, model, cache, device=device))

    return models

class Model:
    def __init__(self, 
                 source_tokenizer,
                 target_tokenizer,
                 model,
                 cache: Optional[bool] = False,
                 device: torch.device = torch.device("cpu"),
                 **kwargs):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.model = model

        self.model_kwargs = self.model.config # self.model_kwargs.attr vs self.model_kwargs['attr]
        self.model_kwargs.update(kwargs)
        self.model_kwargs.use_cache = cache

        self.cache = cache
        self.device = device

        self.num_beams = None
        self.batch_size = None
        self.batch_beam_size = None

        self.pad_token_id = self.target_tokenizer.pad_token_id
        self.eos_token_ids = self.target_tokenizer.eos_token_id

        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]

        self.is_encoder_decoder = hasattr(self.model, "config") and hasattr(self.model.config, "is_encoder_decoder") and self.model.config.is_encoder_decoder

        self.output_attentions = False
        self.output_hidden_states = False

        self.encoder_attentions = None
        self.encoder_hidden_states = None
        self.encoder_outputs = None

        # used to hold the generated tokens
        self.output_ids = None
        self.beam_scores = None

        self.bad_words = [_ for _ in target_tokenizer.added_tokens_decoder.keys() if _ not in [target_tokenizer.eos_token_id]] \
            + [_ for _ in range(len(target_tokenizer.get_vocab()), model.config.vocab_size)]
        
        # init values
        self.stopping_criteria = None
        self.decoder_prompt_len = None


        # SOME STUFF TO MAKE DETOKENIZATION FASTER
        self.skip_special_vocab_itos = [_[0] if (_[1] not in self.bad_words and not self.is_eos(_[1])) else "" for _ in sorted(self.target_tokenizer.get_vocab().items(), key=lambda kv: kv[1])]
        self.vocab_itos = [_[0] for _ in sorted(self.target_tokenizer.get_vocab().items(), key=lambda kv: kv[1])]

        self.speed_vocab = True
        self.vocab_size = len(self.skip_special_vocab_itos)

        if self.is_encoder_decoder:
            self.legacy_added_tokens = set(self.target_tokenizer.added_tokens_decoder.keys()) - set(self.target_tokenizer.all_special_tokens) | {
            token for token in self.target_tokenizer.additional_special_tokens if self.target_tokenizer.convert_tokens_to_ids(token) >= self.target_tokenizer.vocab_size
        }
        else:
            self.legacy_added_tokens = set()


    def set_input(self,
                    batch: str,
                    num_beams: int,
                    max_length: int,
                ):
        """
        Tokenize and encode the input. For seq2seq models, store the encoder outputs
        for passing in the generation loop in step().
        """
        if self.is_encoder_decoder:
            bos_tokens = [[] for _ in batch]
            for batch_i, item in enumerate(batch):

                # BOS tokens are *already* tokenized special ids
                # There may be a variant of this going forward where we pass the ids instead
                logger.debug(f"Tokenizing encoder inputs for batch {batch_i}")
                bos_tokens[batch_i] = tokenize(self.source_tokenizer, bos_tokens=item.get('encoder_bos_tokens', None), inputs=item.get('encoder_inputs', None))

                # You must pass either BOS tokens or encoder inputs
                assert len(bos_tokens[batch_i]) > 0, "No encoder inputs provided"

            # pack and pad these encoder inputs
            logger.debug(f"Encoder inputs: {bos_tokens}")
            self.input_ids = pad_sequence([torch.tensor(_) for _ in bos_tokens], batch_first=True, padding_value=self.source_tokenizer.pad_token_id)  
            input_ids = self.input_ids.repeat_interleave(num_beams, dim=0).to(self.device)
            encoder_attention_mask = (input_ids != self.pad_token_id).int().to(self.device)
            self.encoder_outputs = self.model.get_encoder()(
                input_ids,
                attention_mask = encoder_attention_mask,
                return_dict=True,
                output_attentions = True,
                output_hidden_states = True,
            )

            self.model_kwargs.encoder_outputs = self.encoder_outputs
            self.model_kwargs.attention_mask = encoder_attention_mask
            self.encoder_attentions = self.encoder_outputs['attentions']
            self.encoder_hidden_states = (self.encoder_outputs['hidden_states'])
        # else:
        #     bos_tokens = [[self.model.config.bos_token_id] for _ in batch]
        #     self.input_ids = pad_sequence([torch.tensor(_) for _ in bos_tokens], batch_first=True, padding_value=self.model.config.pad_token_id)  
        #     input_ids = self.input_ids.repeat_interleave(num_beams, dim=0).to(self.device)

        # Now we set up the rest of the model and the decoder
        self.batch_size = len(batch)
        self.num_beams = num_beams
        self.batch_beam_size = self.batch_size * self.num_beams

        # Initialize the scores of the beams
        self.beam_scores = torch.zeros((self.batch_size, num_beams), dtype=torch.float, device=self.device)
        # For the first step, only one beam is valid--set all other so that they are never chosen
        self.beam_scores[:, 1:] = -1e9
        # Flatten so beams are in batch_beam format
        self.beam_scores = self.beam_scores.view((self.batch_size * num_beams,))
        self.beam_token_scores = [[] for _ in range(self.batch_beam_size)]
                
        start_token_id = self.model.config.decoder_start_token_id if self.is_encoder_decoder else self.model.config.bos_token_id
        self.decoder_tokens = [[start_token_id] for _ in range(self.batch_beam_size)]
        
        self.generated_tokens = [[] for _ in range(self.batch_beam_size)]


        # Now we need to initialize the decoder
        # This could be the prompt for language modeling
        # Or it could be the target language token for multilingual MT systems
        # It follows the same structure as the encoder inputs
        for batch_i, item in enumerate(batch):

            logger.debug(f"Tokenizing decoder inputs for batch {batch_i}")
            bos_tokens = tokenize(self.target_tokenizer, bos_tokens=item.get('decoder_bos_tokens', None), inputs=item.get('decoder_inputs', None))
            logger.debug(f"Decoder inputs: {bos_tokens}")
            # We must initialize each beam for each item in the batch
            offset = batch_i * self.num_beams
            for beam_j in range(self.num_beams):
                self.decoder_tokens[offset + beam_j] += bos_tokens
                
        # Eventually we may need to figure out if left padding vs right padding is meaningful
        padding_value = self.pad_token_id

        self.output_ids = pad_sequence([torch.tensor(_) for _ in self.decoder_tokens], batch_first=True, padding_value=padding_value).to(self.device)


    def step(self):
        step_inputs = self.prepare_inputs_for_generation(self.output_ids, **vars(self.model_kwargs))
        step_outputs = self.model(
            **step_inputs,
            return_dict=True,
        )

        # Get the logits for the next token
        # torch selection for BATCH X SEQUENCE LENGTH X VOCABULARY
        # torch.arange will create a index for each batch index position
        # then we select the last non pad item in each sequence
        # last item is length of sequence -1 to account for 0-indexing
        
        # To make the code look better, I changed self.pad_token_id (which is self.target_tokenizer.pad_token_id) to padding_value (which is self.model.config.pad_token_id)
        if self.model.config.pad_token_id:
            padding_value = self.model.config.pad_token_id
        else:
            padding_value = self.model.config.eos_token_id
        if self.is_encoder_decoder:
            next_token_logits = step_outputs.logits[torch.arange(self.batch_beam_size), ((step_inputs['decoder_input_ids'] != padding_value).sum(dim=1)-1)]
        else:
            next_token_logits = step_outputs.logits[torch.arange(self.batch_beam_size), ((step_inputs['input_ids'] != padding_value).sum(dim=1)-1)]

        # Massage the logits. This is how prefix decoding is enforced.
        next_token_logits[:, self.bad_words] = -float("inf")
        next_token_scores = torch.nn.functional.log_softmax(
            next_token_logits, dim=-1
        )
        return step_outputs, next_token_scores
    
    def update_beam(self, beam_idx, beam_next_tokens, beam_scores, step_outputs, debug=False):
        """
        Updates the model's beam sequences by extended the specified beam indices with the selected tokens.
        If {step_outputs} is defined, it will update the cache, as well.

        :param beam_next_tokens: The next tokens to append to the beam sequences
        :param beam_idx: The beam indices that are being extended
        :param beam_scores: The beam scores of each item, to add to the running totals
        :param step_outputs: The model outputs from the current step
        """
        beam_scores = torch.tensor(beam_scores, device=self.device, dtype=torch.float)
        token_scores = [beam_scores[idx] - self.beam_scores[beam_idx[idx]] for idx in range(len(beam_idx))]

        logger.debug(f"Updating beam {beam_idx} with tokens {beam_next_tokens} and scores {beam_scores}")

        # Update the output_ids tensor with the new tokens
        next_beam_tokens = []
        next_generated_tokens = []
        next_token_scores = []
        for beam_i, token, score in zip(beam_idx, beam_next_tokens, token_scores):

            # If the generated token is not a pad token, add it to the list
            if token != self.pad_token_id:
                next_beam_tokens.append(self.decoder_tokens[beam_i] + [token])
                next_generated_tokens.append(self.generated_tokens[beam_i] + [token])
                next_token_scores.append(self.beam_token_scores[beam_i] + [score])

            # Otherwise, we'll keep the old list without the new addition
            else:
                next_beam_tokens.append(self.decoder_tokens[beam_i])
                next_generated_tokens.append(self.generated_tokens[beam_i])
                next_token_scores.append(self.beam_token_scores[beam_i])

        self.beam_scores = beam_scores
        self.decoder_tokens = next_beam_tokens
        self.generated_tokens = next_generated_tokens
        self.beam_token_scores = next_token_scores
        
        if self.model.config.pad_token_id:
            padding_value = self.model.config.pad_token_id
        else:
            padding_value = self.model.config.eos_token_id
        self.output_ids = pad_sequence([torch.tensor(_) for _ in self.decoder_tokens], batch_first=True, padding_value=padding_value).to(self.device)

        # This is where we'll save the state values for caching
        # Truncation is a hacky solution to not having generation beams being the same length
        if self.cache and step_outputs is not None:
            step_outputs['past_key_values'] = self.truncate_past_key_values(step_outputs['past_key_values'], torch.min((self.output_ids != self.pad_token_id).sum(dim=1)-1))
            self.model_kwargs = self._update_model_kwargs_for_generation(
                step_outputs, self.model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if self.model_kwargs["past_key_values"] is not None:
                self.model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    self.model_kwargs["past_key_values"], torch.tensor(beam_idx, device=self.device)
                )

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
        
        if self.pad_token_id:
            decoder_attention_mask = (self.output_ids != self.pad_token_id).int().to(self.device)
        else:
            decoder_attention_mask = (self.output_ids != self.model.config.eos_token_id).int().to(self.device)
            
        if self.is_encoder_decoder:
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
        else:
            return {
                "input_ids": self.output_ids,  # encoder_outputs is defined. input_ids not needed
                "past_key_values": past_key_values,
                "attention_mask": decoder_attention_mask,
                "head_mask": decoder_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
    
    def is_eos(self, token):
        return int(token) in self.eos_token_ids


    def get_beam_string(self, step_i, model_i):
        out = ["======================================================================="]
        for beam_idx in range(self.batch_beam_size):
            out.append(f"BATCH {beam_idx // self.num_beams}\tBEAM {beam_idx % self.batch_size}\tMODEL {model_i}")
            tokens = self.target_tokenizer.convert_ids_to_tokens(self.output_ids[beam_idx].tolist())
            token_str = self.target_tokenizer.decode(self.output_ids[beam_idx], skip_special_tokens=True)
            out.append(f"len={len(tokens)} {self.beam_scores[beam_idx]} {' '.join(tokens)} {self.output_ids[beam_idx]} {token_str}")
        out.append("=======================================================================")
        return out
    



    ######################## EXTRA GARBAGE TO MAKE TOKENIZATION FASTER ############################
    

    def extend_beam_string(self, beam_index, token_id):
        """
        Extends the beam string with the specified token.
        """
        if token_id in [2, 22]:
            pass
        sequence = self.generated_tokens[beam_index] + [token_id]
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
            return self.target_tokenizer.convert_ids_to_tokens([token_id])[0]


    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        # from this file https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/tokenization_utils.py#L405
        tokenizer = self.target_tokenizer
        tokenizer._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # Convert the token IDs to strings
        filtered_tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id, skip_special_tokens=skip_special_tokens)
            if len(token) > 0:
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
                    string = tokenizer.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))
            # this is a dumb thing
            trailing_spaces = 0
            for c in current_sub_text[::-1]:
                if c == "▁":
                    trailing_spaces += 1
                else:
                    break
            # if len(tokenizer.convert_tokens_to_string(current_sub_text[-1:])) == 0:
            #     sub_texts[-1] += " "
            sub_texts[-1] += " " * trailing_spaces

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
        