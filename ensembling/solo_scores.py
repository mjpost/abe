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
        
    # token_scores = []
    # for id, logit in zip(decoder_input[0][1:], all_logits):
    #     token_scores.append(logit[id].item())

    return all_logits


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


