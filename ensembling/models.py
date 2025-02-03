
########################################################################################################
#                                                                                                      #
#                                         PACKAGING AND LOGGING                                        #
#                                                                                                      #
########################################################################################################

import os, sys
import logging
import pathlib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ensembling")


if (__package__ is None or __package__ == "")  and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'ensembling'

########################################################################################################
#                                                                                                      #
#                                               IMPORTS                                                #
#                                                                                                      #
########################################################################################################


from typing import Optional, List, Union

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM
import torch
from torch.nn.utils.rnn import pad_sequence

from ensembling.utils import tokenize, Trie, TOKENIZER_CONFIG, BYTE_MAP


########################################################################################################
#                                                                                                      #
#                                   MODEL AND TOKENIZER CLASSES                                        #
#                                                                                                      #
########################################################################################################

class Model:
    def __init__(self, 
                 source_tokenizer: AutoTokenizer,
                 target_tokenizer: AutoTokenizer,
                 model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
                 cache: Optional[bool] = False,
                 device: torch.device = torch.device("cpu"),
                 **kwargs):
        
        # in the future, we made need logic for types of tokenizers
        # https://huggingface.co/docs/tokenizers/api/decoders
        self.model_name = model.config.name_or_path
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        # we are going to maintain a separate tokenizer that does not clean up whitespace
        self.whitespace_tokenizer = FastTokenizer(target_tokenizer,
                                                  lstrip = TOKENIZER_CONFIG.get(self.model_name, {}).get("lstrip", False),
                                                    special_character = TOKENIZER_CONFIG.get(self.model_name, {}).get("special_character", '\u2581'),
                                                    begin_word = TOKENIZER_CONFIG.get(self.model_name, {}).get("begin_word", True),
                                                    byte_map = TOKENIZER_CONFIG.get(self.model_name, {}).get("byte_map", BYTE_MAP),
                                                    add_space = TOKENIZER_CONFIG.get(self.model_name, {}).get("add_space", True)
                                                )

        self.model = model

        self.model_kwargs = self.model.config # self.model_kwargs.attr vs self.model_kwargs['attr]
        self.model_kwargs.update(kwargs)
        self.model_kwargs.use_cache = cache

        self.cache = cache
        self.device = device

        self.num_beams = None
        self.batch_size = None
        self.batch_beam_size = None

        self.pad_token_id = self.target_tokenizer.pad_token_id if self.target_tokenizer.pad_token_id is not None else self.target_tokenizer.eos_token_id
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

        self.trie = None


    def set_input(self,
                    batch: List[dict],
                    num_beams: int,
                    max_length: int,
                    sample = False
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
                bos_tokens[batch_i] = tokenize(self.source_tokenizer, bos_tokens=item.get('encoder_bos_tokens', None), inputs=item.get('encoder_inputs', None), eos=True)

                # You must pass either BOS tokens or encoder inputs
                assert len(bos_tokens[batch_i]) > 0, "No encoder inputs provided"

            # pack and pad these encoder inputs
            logger.debug(f"Encoder inputs: {bos_tokens}")
            self.input_ids = pad_sequence([torch.tensor(_) for _ in bos_tokens], batch_first=True, padding_value=self.source_tokenizer.pad_token_id)
            # TODO: There should be some checking here for max length
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

        # Now we set up the rest of the model and the decoder
        self.batch_size = len(batch)
        self.num_beams = num_beams
        self.batch_beam_size = self.batch_size * self.num_beams

        # Initialize the scores of the beams
        self.beam_scores = torch.zeros((self.batch_size, num_beams), dtype=torch.float, device=self.device)

        # For the first step, only one beam is valid--set all other so that they are never chosen
        if not sample:
            self.beam_scores[:, 1:] = -1e9

        # Flatten so beams are in batch_beam format
        self.beam_scores = self.beam_scores.view((self.batch_size * num_beams,))
        self.beam_token_scores = [[] for _ in range(self.batch_beam_size)]
                
        start_token_id = self.model.config.decoder_start_token_id if self.is_encoder_decoder else self.model.config.bos_token_id
        self.decoder_tokens = [[start_token_id] for _ in range(self.batch_beam_size)]
        
        self.generated_tokens = [[] for _ in range(self.batch_beam_size)]
        self.byte_strings = ["".encode('utf-8') for _ in range(self.batch_beam_size)]

        # Now we need to initialize the decoder
        # This could be the prompt for language modeling
        # Or it could be the target language token for multilingual MT systems
        # It follows the same structure as the encoder inputs
        for batch_i, item in enumerate(batch):
            logger.debug(f"Tokenizing decoder inputs for batch {batch_i}")
            bos_tokens = tokenize(self.target_tokenizer, bos_tokens=item.get('decoder_bos_tokens', None), inputs=item.get('decoder_inputs', None), eos=False)
            logger.debug(f"Decoder inputs: {bos_tokens}")
            # We must initialize each beam for each item in the batch
            offset = batch_i * self.num_beams
            for beam_j in range(self.num_beams):
                self.decoder_tokens[offset + beam_j] += bos_tokens

        self.output_ids = pad_sequence([torch.tensor(_) for _ in self.decoder_tokens], batch_first=True, padding_value=self.pad_token_id).to(self.device)


    def step(self):
        step_inputs = self.prepare_inputs_for_generation(self.output_ids, **vars(self.model_kwargs))
        with torch.no_grad():
            step_outputs = self.model(
                **step_inputs,
                return_dict=True,
            )

        if self.is_encoder_decoder:
            next_token_logits = step_outputs.logits[torch.arange(self.batch_beam_size), (self.build_attention_mask(step_inputs['decoder_input_ids']).sum(dim=1) - 1)]
        else:
            next_token_logits = step_outputs.logits[torch.arange(self.batch_beam_size), (self.build_attention_mask(step_inputs['input_ids']).sum(dim=1) - 1)]

        # Massage the logits. This is how prefix decoding is enforced.
        next_token_logits[:, self.bad_words] = -float("inf")
        next_token_scores = torch.nn.functional.log_softmax(
            next_token_logits, dim=-1
        )

        return step_outputs, next_token_scores
    
    def update_beam(self, 
                    beam_idx: int, 
                    beam_next_tokens: List[int], 
                    beam_scores: torch.tensor,
                    step_outputs):
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
        next_byte_strings = []
        for beam_i, token, score in zip(beam_idx, beam_next_tokens, token_scores):

            # If the generated token is not a pad token, add it to the list
            if token != -1:
                next_beam_tokens.append(self.decoder_tokens[beam_i] + [token])
                next_generated_tokens.append(self.generated_tokens[beam_i] + [token])
                next_token_scores.append(self.beam_token_scores[beam_i] + [score])
                next_byte_strings.append(self.byte_strings[beam_i] + self.whitespace_tokenizer.decode(token))

            # Otherwise, we'll keep the old list without the new addition
            else:
                next_beam_tokens.append(self.decoder_tokens[beam_i])
                next_generated_tokens.append(self.generated_tokens[beam_i])
                next_token_scores.append(self.beam_token_scores[beam_i])
                next_byte_strings.append(self.byte_strings[beam_i])

        self.beam_scores = beam_scores
        self.decoder_tokens = next_beam_tokens
        self.generated_tokens = next_generated_tokens
        self.beam_token_scores = next_token_scores
        self.byte_strings = next_byte_strings
        
        self.output_ids = pad_sequence([torch.tensor(_) for _ in self.decoder_tokens], batch_first=True, padding_value=self.pad_token_id).to(self.device)

        # This is where we'll save the state values for caching
        # Truncation is a hacky solution to not having generation beams being the same length
        if self.cache and step_outputs is not None:
            step_outputs['past_key_values'] = self.truncate_past_key_values(step_outputs['past_key_values'], torch.min((self.build_attention_mask(self.output_ids)).sum(dim=1)-1))
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


    def build_attention_mask(self, input_ids):
        true_mask = (input_ids == input_ids)
        for i, j in zip(true_mask, self.decoder_tokens):
            i[len(j):] = False
        return true_mask.int().to(self.device)


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
        
        decoder_attention_mask = self.build_attention_mask(decoder_input_ids)

            
        if self.is_encoder_decoder:
            # We are no longer using `head mask` because it is not compatible with all types of models?
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
        else:
            return {
                "input_ids": self.output_ids,  # encoder_outputs is defined. input_ids not needed
                "past_key_values": past_key_values,
                "attention_mask": decoder_attention_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
    
    def is_eos(self, token):
        return int(token) in self.eos_token_ids

    def get_logging_string(self, step_i, model_i, type="BEAM"):
        out = ["======================================================================="]
        for beam_idx in range(self.batch_beam_size):
            out.append(f"BATCH {beam_idx // self.num_beams}\t{type} {beam_idx % self.batch_size}\tMODEL {model_i}")
            tokens = self.target_tokenizer.convert_ids_to_tokens(self.output_ids[beam_idx].tolist())
            token_str = self.target_tokenizer.decode(self.output_ids[beam_idx], skip_special_tokens=True).replace('\n', ' \\n ')
            out.append(f"len={len(tokens)} {self.beam_scores[beam_idx]} {' '.join(tokens)} {self.output_ids[beam_idx]} {token_str}")
        out.append("=======================================================================")
        return out


    ######################## EXTRA GARBAGE TO MAKE TOKENIZATION FASTER ############################
    

    def extend_beam_string(self, beam_index, token_id):
        """
        Extends the beam string with the specified token.
        """
        return self.whitespace_tokenizer.extend_beam_string(self.byte_strings[beam_index], token_id)

    def id_to_token(self, token_id, skip_special_tokens=False):
        search_in = self.vocab_itos if not skip_special_tokens else self.skip_special_vocab_itos
        if token_id == -1:
            return "<pad>" if not skip_special_tokens else ""
        if self.speed_vocab:
            if self.vocab_size > token_id:
                return search_in[token_id]
            else:
                return "<unk>" if not skip_special_tokens else ""
        else:
            return self.target_tokenizer.convert_ids_to_tokens([token_id])[0]


class FastTokenizer():
    def __init__(self, original_tokenizer, lstrip=False, special_character='\u2581', begin_word=True, add_space=True, byte_map=BYTE_MAP):
        self.vocab = []
        for tok, tok_id in sorted(original_tokenizer.get_vocab().items(), key=lambda kv: kv[1]):
            if tok_id not in original_tokenizer.all_special_ids and tok not in byte_map:
                byte_tok = ""
                if add_space:
                    if begin_word and tok.startswith(special_character):
                        byte_tok += " "
                    elif (not begin_word) and (not tok.startswith(special_character)):
                        byte_tok += " "
                byte_tok += original_tokenizer.decode(tok_id)
                self.vocab.append(byte_tok.encode('utf-8'))
            elif tok in byte_map:
                self.vocab.append(bytes([byte_map.index(tok)]))
            elif tok_id == original_tokenizer.unk_token_id:
                self.vocab.append(tok.encode('utf-8'))
            else:
                self.vocab.append("".encode('utf-8'))
        self.lstrip = lstrip

    def decode(self, token_id):
        if token_id == -1:
            return "".encode('utf-8')
        if token_id > len(self.vocab):
            return "<unk>".encode('utf-8')
        return self.vocab[token_id]
    
    def extend_beam_string(self, beam_string, token_id):
        bytes = beam_string + self.decode(token_id)
        if self.lstrip:
            bytes = " ".encode('utf-8') + bytes
        return bytes
    

########################################################################################################
#                                                                                                      #
#                                   EXTERNAL FACING BUILDERS                                           #
#                                                                                                      #
########################################################################################################


def build_tries(models: List[Model]):
    for model_i, model in enumerate(models):
        logger.info(f"Building trie for model {model_i}: {model.model_kwargs.name_or_path}")
        vocab_length = model.model_kwargs.vocab_size
        model.trie = Trie(vocab_length)
        for tok_id, tok in enumerate(model.whitespace_tokenizer.vocab):
            model.trie.add_string(tok, tok_id)


def get_models(model_names, device = torch.device('cpu'), cache : bool = False, half : bool = False):
    models = []

    for model_i, model_name in enumerate(model_names):
        print(f"Instantiating model {model_name}", file=sys.stderr)
        try:
            config = AutoConfig.from_pretrained(model_name)
            source_tokenizer = AutoTokenizer.from_pretrained(model_name)
            target_tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.bfloat16 if half else torch.float32
            if config.is_encoder_decoder:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        except Exception as e:
            logger.error(f"Error instantiating model {model_name}: {e}")
            sys.exit(-1)

        model.eval().to(device)

        models.append(Model(source_tokenizer, target_tokenizer, model, cache, device=device))

    return models