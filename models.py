from typing import Optional, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    ForcedBOSTokenLogitsProcessor,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
)

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
        bos_token_id = tokenizer.lang_code_to_id["fr"]

        return Model(model=model, tokenizer=tokenizer, bos_force_token=bos_token_id)
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
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_force_token = bos_force_token
        self.kwargs = kwargs

        self.output_attentions = False
        self.output_hidden_states = False

        if self.bos_force_token is not None:
            self.logits_processor.append(
                ForcedBOSTokenLogitsProcessor(bos_force_token)
        )
            
        self.input = None

    def set_input(self, line: str, return_tensors="pt"):
        self.input = self.tokenizer(line, return_tensors=return_tensors)
        return self.input.input_ids
    
    def prepare_inputs_for_generation(self, inputs, model_kwargs):
        return self.model.prepare_inputs_for_generation(inputs, **model_kwargs)

    def step(self, model_inputs):
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )
        return outputs
