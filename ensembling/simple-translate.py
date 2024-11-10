import argparse
import sys
import torch
import time

def load_model(model_name, source_language, target_language):
    if model_name in ["facebook/m2m100_418M", "facebook/m2m100_1.2B"]:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, 
                                                    src_lang=source_language, 
                                                    tgt_lang=target_language)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        bos_force_token = tokenizer.convert_tokens_to_ids([f"__{target_language}__"])[0]

    elif model_name in ["facebook/nllb-200-distilled-600M"]:
        from lang_map import NLLB_LANGMAP
        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

        tokenizer = NllbTokenizer.from_pretrained(model_name, 
                                                  src_lang=NLLB_LANGMAP[source_language], 
                                                  tgt_lang=NLLB_LANGMAP[target_language])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        bos_force_token = tokenizer.convert_tokens_to_ids([NLLB_LANGMAP.get(target_language, target_language)])[0]

    return model, tokenizer, bos_force_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--target_lang", type=str, default="fr")
    parser.add_argument("--input", "-i", default=None, type=str)
    parser.add_argument("--output", "-o", default=None, type=str)
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--time", default=False, action="store_true")

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')

    model, tokenizer, force_bos_token = load_model(args.model_name, args.source_lang, args.target_lang)
    model = model.to(device)

    istream = open(args.input, "r") if args.input is not None else sys.stdin
    ostream = open(args.output, "w") if args.output is not None else sys.stdout

    for line in istream:
        line = line.rstrip()
        encoded_en = tokenizer(line, return_tensors="pt")
        encoded_en = encoded_en.to(device)
        start = time.time()
        generated_tokens = model.generate(**encoded_en, 
                                          forced_bos_token_id=force_bos_token, 
                                          use_cache=False)
        end = time.time()
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        if args.time:
            print(f"TIME: {end - start}\t{result}", file=ostream)
        else:
            print(result, file=ostream)
