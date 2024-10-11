import argparse
import sys
import torch
import time
import json

from transformers import AutoModelWithLMHead

def load_model(model_name, device):
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return model.eval().to(device)

def get_scores(line, model, device=torch.device('cpu')):
    encoder_input = torch.tensor([line["inputs"]]).to(device) # token ids
    decoder_input = torch.tensor([line["outputs"]]).to(device) # token ids

    outputs = model(input_ids=encoder_input, decoder_input_ids=decoder_input).logits[0]
    outputs = torch.nn.functional.log_softmax(outputs, dim=-1)

    token_scores = []
    for id, logits in zip(decoder_input[0][1:], outputs[:-1]):
        token_scores.append(logits[id].item())

    return token_scores

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')

    model = load_model(args.model_name, device)

    istream = open(args.input, "r") if args.input is not None else sys.stdin
    ostream = open(args.output, "w") if args.output is not None else sys.stdout

    for line in istream:
        line = json.loads(line.strip())

        token_scores = get_scores(line, model, device)

        print(json.dumps({"scores": token_scores}), file=ostream)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--input", "-i", default=None, type=str)
    parser.add_argument("--output", "-o", default=None, type=str)
    parser.add_argument("--cpu", default=False, action="store_true")

    args = parser.parse_args()

    main(args)


