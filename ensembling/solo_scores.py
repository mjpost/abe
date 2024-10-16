import argparse
import sys
import torch
import time
import json

from transformers import AutoModelWithLMHead
import subprocess

def load_model(model_name, device):
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return model.eval().to(device)

def get_scores(line, device=torch.device('cpu')):
    encoder_inputs = line["input_ids"] # token ids
    decoder_inputs = line["token_ids"] # token ids

    token_scores_dict = {0: [], 1: []}
    for i in range(len(encoder_inputs)):
        print("Model", args.models[i])
        model = AutoModelWithLMHead.from_pretrained(args.models[i])
        input_ids = torch.tensor(encoder_inputs[i]).unsqueeze(0).to(device)
        
        decoder_input_ids = torch.tensor(decoder_inputs[i]).unsqueeze(0).to(device)
        

        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits[0]
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)

        token_scores = []
        for id, logits in zip(decoder_input_ids[0][1:], outputs[:-1]):
            token_scores.append(logits[id].item())
        
        token_scores_dict[i] = token_scores
    return token_scores_dict

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    istream = open(args.input, "r") if args.input is not None else sys.stdin
    ostream = open(args.output, "w") if args.output is not None else sys.stdout

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
    parser.add_argument("--models", '-m', type=str, help='Models to ensemble', nargs='+', default=["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"])
    parser.add_argument("--input", "-i", default=None, type=str)
    parser.add_argument("--output", "-o", default=None, type=str)
    parser.add_argument("--cpu", default=False, action="store_true")

    args = parser.parse_args()

    main(args)


