#!/usr/bin/env python3

import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main(args):
    # m2m100_1.2B
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    for line in sys.stdin:
        line = line.rstrip()
        inputs = tokenizer(line, return_tensors="pt")

        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[args.target_lang], max_length=args.max_output_tokens
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

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