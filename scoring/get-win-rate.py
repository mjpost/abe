import argparse
import sys
import torch
import time
import json
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import subprocess


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

def get_scores(line, model, bad_words, device=torch.device('cpu')):
    encoder_input = torch.tensor([line["input_ids"]]).to(device) # token ids
    decoder_input = torch.tensor([line["output_ids"]]).to(device) # token ids

    all_logits = []
    for i in range(decoder_input.shape[-1]-1):
        outputs = model(
                        input_ids=encoder_input,
                        decoder_input_ids=decoder_input[:, :i+1],
                        return_dict=True,
                        output_hidden_states=True)
        logits = outputs.logits[0]
        logits[:, bad_words] = -float("inf")
        logits = torch.nn.functional.log_softmax(logits, dim=-1)[-1]
        all_logits.append(
             logits[decoder_input[0, i+1]].item()
        )

    return all_logits




def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')


    model_one = args.models[0]
    model_two = args.models[1]

    clean_model_one = '-'.join(model_one.split('/')[1:])
    clean_model_two = '-'.join(model_two.split('/')[1:])

    ensemble_name = f"{clean_model_one}+{clean_model_two}"
    ensemble_file = open(os.path.join(args.ensemble_dir, ensemble_name))
    
    model_one_file = open(os.path.join(args.individual_dir, clean_model_one))
    model_two_file = open(os.path.join(args.individual_dir, clean_model_two))

    for eline, m1line, m2line in zip(ensemble_file, model_one_file, model_two_file):


    for line in istream:
        line = json.loads(line.strip())
        token_scores_dict = get_scores(line, device)
        sequence_scores = {key: sum(value[1:]) for key, value in token_scores_dict.items()}
        weights = line['weights']
        ensemble_score_calculated = sum([weights[i] * sequence_scores[i] for i in range(len(sequence_scores))])
        if abs(ensemble_score_calculated - line['combined_score']) > 0.01:
            print("Ensemble score mismatch", ensemble_score_calculated, line['combined_score'])
        else:
            print("Ensemble score matched", abs(ensemble_score_calculated-line['combined_score']))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", "-m", type=str, nargs='+', help="Models to test")
    parser.add_argument("--inputs", "-i", type=str, nargs='+', help="Input file")
    parser.add_argument("--ensemble_dir", "-e", type=str, default = "/exp/rwicks/ensemble24/translations/wmt24/en-de/targets", help="Directory containing ensemble files")
    parser.add_argument("--individual_dir", "-i", type=str, default = "/exp/rwicks/ensemble24/simple-translations/outputs/targets", help="Directory containing individual model files")
    parser.add_argument("--cpu", default=False, action="store_true")

    args = parser.parse_args()

    main(args)


