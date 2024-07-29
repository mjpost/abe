import argparse
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import torch
import sys


def get_m2m_scores(tokenizer, model, source, reference, source_language="en", target_language="fr"):
    source_ids = torch.tensor([[tokenizer.lang_code_to_id[source_language]] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source)) + [2]])
    reference_ids = torch.tensor([[2] + [tokenizer.lang_code_to_id[target_language]] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(reference)) + [2]])

    input_logprobs = []
    logits = model(input_ids=source_ids, decoder_input_ids=reference_ids).logits

    all_tokens_logprobs = torch.log_softmax(logits.double(), dim=2)
    for k in range(1, reference_ids.shape[1]-2):
        input_logprobs.append(all_tokens_logprobs[:, k-1, reference_ids[0,k]].item())
    return sum(input_logprobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unit tests")
    parser.add_argument("--model", default="facebook/m2m100_418M", help="Name of huggingface model")
    parser.add_argument("--src", default="en", help="Source language")
    parser.add_argument("--tgt", default="fr", help="Target language")
    parser.add_argument("--input", default=None, help="Input stream. default stdin")
    args = parser.parse_args()


    tokenizer = M2M100Tokenizer.from_pretrained(args.model, src_lang=args.src, tgt_lang=args.tgt)
    model = M2M100ForConditionalGeneration.from_pretrained(args.model)

    istream = sys.stdin if args.input is None else open(args.input)

    for line in istream:
        source, reference = line.strip().split("\t")
        score = get_m2m_scores(tokenizer, model, source, reference, args.src, args.tgt)
        print(score)