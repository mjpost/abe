import logging

import argparse
import sys
import torch
import json

from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def load_model(model_name, half=False):
    config = AutoConfig.from_pretrained(model_name)
    source_tokenizer = AutoTokenizer.from_pretrained(model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if half else torch.float32
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model, source_tokenizer, target_tokenizer

def tokenize(tokenizer, bos_tokens=None, inputs=None, eos=True):
    out = []
    if bos_tokens is not None:
        out += tokenizer.convert_tokens_to_ids(bos_tokens)
        if tokenizer.bos_token_id not in out and tokenizer.unk_token_id in out:
            logging.error(f"UNK token found in BOS tokens")
            sys.exit(-1)
        logging.debug(f"ids for BOS: {out}")
    if inputs is not None:
        if eos:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs) + [tokenizer.eos_token])
        else:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs))
        out += tokens
        logging.debug(f"Tokenized input: {tokens}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--half", default=False, action="store_true")
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--input", "-i", default=None, type=str)
    parser.add_argument("--output", "-o", default=None, type=str)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')

    model, source_tokenizer, target_tokenizer = load_model(args.model_name, half = args.half)
    model = model.to(device)

    istream = open(args.input, "r") if args.input is not None else sys.stdin
    ostream = open(args.output, "w") if args.output is not None else sys.stdout

    for line in istream:
        item = json.loads(line.strip())

        
        if model.config.is_encoder_decoder:
            encoder_inputs = tokenize(source_tokenizer, bos_tokens=item.get('encoder_bos_tokens', None), inputs=item.get('encoder_inputs', None), eos=True)
        decoder_inputs = tokenize(target_tokenizer, bos_tokens=item.get('decoder_bos_tokens', None), inputs=item.get('decoder_inputs', None), eos=False)

        if model.config.is_encoder_decoder:
            generated_tokens = model.generate(
                input_ids=torch.tensor([encoder_inputs]).to(device),
                attention_mask = torch.ones((1, len(decoder_inputs))).to(device),
                decoder_input_ids=torch.tensor([decoder_inputs]).to(device),
                use_cache=True,
                max_new_tokens = 256,
                pad_token_id=target_tokenizer.pad_token_id
            )
        else:
            generated_tokens = model.generate(
                input_ids=torch.tensor([decoder_inputs]).to(device),
                attention_mask = torch.ones((1, len(decoder_inputs))).to(device),
                use_cache = True,
                max_new_tokens = 256,
                pad_token_id=target_tokenizer.eos_token_id
            )
        generated_tokens = generated_tokens[:, len(decoder_inputs):]
        # generated_tokens = generated_tokens[0]
        
        # encoded_en = tokenizer(line, return_tensors="pt")
        # encoded_en = encoded_en.to(device)
        # generated_tokens = model.generate(**encoded_en, 
        #                                   forced_bos_token_id=force_bos_token, 
        #                                   use_cache=False)
        result = target_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].replace('\n', ' ').strip()
        print(result)
        # breakpoint()