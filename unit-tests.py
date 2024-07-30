import argparse
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, \
                        NllbTokenizer, AutoModelForSeq2SeqLM
import torch
import math
import sys


LANG_MAP = {
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn"
}

# class Scorer():
#     def __init__(self, model, tokenizer, m2m=True):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.m2m = m2m

#     def __call__(self, source, reference, source_language="en", target_language="fr"):
#         if self.m2m:
#             return self.get_m2m_scores(source, reference, source_language, target_language)
#         else:
#             return self.get_nllb_scores(source, reference, source_language, target_language)

#     def get_m2m_scores(self, source, reference, source_language="en", target_language="fr"):
#         # source_ids = torch.tensor([[self.tokenizer.lang_code_to_id[source_language]] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(source)) + [2]])
#         source_ids = tokenizer(source, return_tensors="pt").input_ids
#         reference_ids = torch.tensor([[2] + [self.tokenizer.lang_code_to_id[target_language]] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(reference)) + [2]])

#         input_logprobs = []
#         logits = self.model(input_ids=source_ids, decoder_input_ids=reference_ids).logits

#         all_tokens_logprobs = torch.log_softmax(logits.double(), dim=2)
#         for k in range(1, reference_ids.shape[1]-2):
#             input_logprobs.append(all_tokens_logprobs[:, k-1, reference_ids[0,k]].item())
#         return sum(input_logprobs)
    

#     def get_scores(self, source, reference, source_language="en", target_language="fr"):
#         source_ids = tokenizer(source, return_tensors="pt").input_ids
#         reference_ids = torch.tensor([[2] + [self.tokenizer.lang_code_to_id[target_language]] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(reference)) + [2]])

#         input_logprobs = []
#         logits = self.model(input_ids=source_ids, decoder_input_ids=reference_ids).logits

#         all_tokens_logprobs = torch.log_softmax(logits.double(), dim=2)
#         for k in range(1, reference_ids.shape[1]-2):
#             input_logprobs.append(all_tokens_logprobs[:, k-1, reference_ids[0,k]].item())
#         return sum(input_logprobs)

def get_scores(model, tokenizer, source, reference, source_language="en", target_language="fr"):
    '''
    Gets the model scores for a given source and reference pair.
    Adds </s>, <lang_code> (prefix), and </s> (suffix) tokens to the reference respectively.
    Returns the sum of log probabilities of the reference tokens given the source.
    '''
    source_ids = tokenizer(source, return_tensors="pt").input_ids
    reference_ids = torch.tensor([[2] + [tokenizer.lang_code_to_id[target_language]] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(reference)) + [2]])

    input_logprobs = []
    logits = model(input_ids=source_ids, decoder_input_ids=reference_ids).logits

    all_tokens_logprobs = torch.log_softmax(logits.double(), dim=2)
    for k in range(1, reference_ids.shape[1]-1):
        input_logprobs.append(all_tokens_logprobs[:, k-1, reference_ids[0,k]].item())
    return sum(input_logprobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unit tests")
    parser.add_argument("--model", 
                        default="facebook/m2m100_418M", 
                        help="Name of huggingface model", 
                        choices=["facebook/m2m100_418M", "facebook/nllb-200-distilled-600M"])
    parser.add_argument("--src", default="en", help="Source language")
    parser.add_argument("--tgt", default="fr", help="Target language")
    parser.add_argument("--input", default=None, help="Input stream. default stdin")
    args = parser.parse_args()

    if args.model == "facebook/nllb-200-distilled-600M":
        '''
        NLLB model uses a Seq2SeqLM model, while M2M100 uses a ConditionalGeneration model. 
        '''
        src = LANG_MAP.get(args.src, args.src)
        tgt = LANG_MAP.get(args.tgt, args.tgt)
        tokenizer = NllbTokenizer.from_pretrained(args.model, src_lang=src, tgt_lang=tgt)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    else:
        src = args.src
        tgt = args.tgt
        tokenizer = M2M100Tokenizer.from_pretrained(args.model, src_lang=src, tgt_lang=tgt)
        model = M2M100ForConditionalGeneration.from_pretrained(args.model)


    istream = sys.stdin if args.input is None else open(args.input)

    for line in istream:
        source, reference = line.strip().split("\t")
        score = get_scores(model, tokenizer, source, reference, src, tgt)
        print(score)