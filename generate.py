#!/usr/bin/env python3

import sys
from typing import Optional, Union, Tuple, Dict, Any, List, Callable
import warnings
import models
import torch
import logging

from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.cache_utils import Cache


logger = logging.get_logger(__name__)



def _get_stopping_criteria(
    generation_config: GenerationConfig, stopping_criteria: Optional[StoppingCriteriaList]
) -> StoppingCriteriaList:
    criteria = StoppingCriteriaList()
    if generation_config.max_length is not None:
        max_position_embeddings = None
        criteria.append(
            MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings,
            )
        )
    if generation_config.max_time is not None:
        criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
    criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
    return criteria


def _get_logits_processor(
    self,
    generation_config: GenerationConfig,
    input_ids_seq_length: int,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    logits_processor: Optional[LogitsProcessorList],
) -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    # instantiate processors list
    processors = LogitsProcessorList

    if (
        generation_config.min_length is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_length > 0
    ):
        processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
    if (
        generation_config.min_new_tokens is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_new_tokens > 0
    ):
        processors.append(
            MinNewTokensLengthLogitsProcessor(
                input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
            )
        )
    if prefix_allowed_tokens_fn is not None:
        processors.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
            )
        )
    if generation_config.forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
    if generation_config.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
        )

    if generation_config.forced_decoder_ids is not None:
        processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids))
    processors = self._merge_criteria_processor_list(processors, logits_processor)
    # `LogitNormalization` should always be the last logit processor, when present
    return processors


def _expand_inputs_for_generation(
    expand_size: int = 1,
    is_encoder_decoder: bool = False,
    input_ids: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

    def _expand_dict_for_generation(dict_to_expand):
        for key in dict_to_expand:
            if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
        return dict_to_expand

    if input_ids is not None:
        input_ids = input_ids.repeat_interleave(expand_size, dim=0)

    model_kwargs = _expand_dict_for_generation(model_kwargs)

    if is_encoder_decoder:
        if model_kwargs.get("encoder_outputs") is None:
            raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

    return input_ids, model_kwargs

_
def _validate_model_kwargs(model_kwargs: Dict[str, Any]):
    """Validates model kwargs for generation. Generate argument typos will also be caught here."""
    # If a `Cache` instance is passed, checks whether the model is compatible with it
    if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
        raise ValueError(
            f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
            "check the model documentation for supported cache formats."
        )

    # Excludes arguments that are handled before calling any model function
    if self.config.is_encoder_decoder:
        for key in ["decoder_input_ids"]:
            model_kwargs.pop(key, None)

    unused_model_args = []
    model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
    # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
    # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
    if "kwargs" in model_args or "model_kwargs" in model_args:
        model_args |= set(inspect.signature(self.forward).parameters)

    # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
    if self.config.is_encoder_decoder:
        base_model = getattr(self, self.base_model_prefix, None)

        # allow encoder kwargs
        encoder = getattr(self, "encoder", None)
        # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
        # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
        # TODO: A better way to handle this.
        if encoder is None and base_model is not None:
            encoder = getattr(base_model, "encoder", None)

        if encoder is not None:
            encoder_model_args = set(inspect.signature(encoder.forward).parameters)
            model_args |= encoder_model_args

        # allow decoder kwargs
        decoder = getattr(self, "decoder", None)
        if decoder is None and base_model is not None:
            decoder = getattr(base_model, "decoder", None)

        if decoder is not None:
            decoder_model_args = set(inspect.signature(decoder.forward).parameters)
            model_args |= {f"decoder_{x}" for x in decoder_model_args}

        # allow assistant_encoder_outputs to be passed if we're doing assisted generating
        if "assistant_encoder_outputs" in model_kwargs:
            model_args |= {"assistant_encoder_outputs"}

    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)

    if unused_model_args:
        raise ValueError(
            f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
            " generate arguments will also show up in this list)"
        )


@torch.no_grad()
def generate(
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which had the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complement the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
        # three conditions must be met
        # 1) the generation config must have been created from the model config (`_from_model_config` field);
        # 2) the generation config must have seen no modification since its creation (the hash is the same);
        # 3) the user must have set generation parameters in the model config.
        generation_config = GenerationConfig.from_model_config(self.config)

    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    _validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length

    # otherwise the total length [inputs-embeds-len + new-tokens-len] will go beyond indicated `max_length``
    elif (
        model_input_name == "inputs_embeds"
        and inputs_tensor.shape[:-1] != input_ids.shape
        and not self.config.is_encoder_decoder
    ):
        generation_config.max_length -= inputs_tensor.shape[1]

    if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if generation_config.cache_implementation == "static":
            if model_kwargs.get("past_key_values", False) is not False:
                raise ValueError(
                    "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
                )
            cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
            if not callable(getattr(self, "_setup_cache", None)):
                raise ValueError(
                    "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                    " Make sure it has a `_setup_cache` function."
                )
            self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    prepared_logits_processor = _get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    # 9. prepare stopping criteria
    prepared_stopping_criteria = _get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    # 10. hard-code beam search

    # 11. prepare beam search scorer
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=generation_config.num_beams,
        device=inputs_tensor.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=generation_config.num_return_sequences,
        max_length=generation_config.max_length,
    )
    # 12. interleave input_ids with `num_beams` additional sequences per batch
    input_ids, model_kwargs = _expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_beams,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )
    # 13. run beam search
    result = self.beam_search(
        input_ids,
        beam_scorer,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        output_scores=generation_config.output_scores,
        output_logits=generation_config.output_logits,
        return_dict_in_generate=generation_config.return_dict_in_generate,
        synced_gpus=False,
        sequential=generation_config.low_memory,
        **model_kwargs,
    )

    return result


def main(args):
    # m2m100_1.2B
    bundle = models.get_model_bundle(args.model_name)
    tokenizer = bundle.tokenizer  # = AutoTokenizer.from_pretrained(args.model_name)
    model = bundle.model  # = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    for line in sys.stdin:
        line = line.rstrip()
        inputs = tokenizer(line, return_tensors="pt")

        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=bundle.bos_force_token, max_length=args.max_output_tokens
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        # print(model.config.decoder_start_token_id, translated_tokens, result, sep="\n")
        print(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, default="facebook/m2m100_418M", help="Model name")
    parser.add_argument("--target-lang", "-t", type=str, default="fr", help="Target language")
    parser.add_argument("--num-beams", "-b", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--noise", "-n", type=float, default=None, help="Add noise to final model logits")
    parser.add_argument("--max-output-tokens", "-l", type=int, default=30, help="Maximum number of output tokens")
    args = parser.parse_args()

    main(args)