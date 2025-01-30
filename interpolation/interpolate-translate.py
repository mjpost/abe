import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

import os

import argparse
import sys
import torch
import torch.nn.functional as F
import random
import math

from transformers import MarianMTModel, AutoTokenizer


class Mixture():
    def __init__(self, ensembled_models, input_ids, encoder_attention_mask, lambdas=[1.0], num_beams=5):

        # ensembled_models: list of models to ensemble
        self.ensembled_models = ensembled_models

        # we will save the encoder outputs so each forward step on the decoder is faster
        self.encoder_outputs = []

        # this extends the inputs to have the same size as the num beams
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)

        # we don't really use the attention mask since there's no padding in the input_ids
        encoder_attention_mask = encoder_attention_mask.repeat_interleave(num_beams, dim=0)

        # then we need to get the encoder_outputs for each model
        for i, model in enumerate(ensembled_models):
            encoder_outputs = model.get_encoder()(
                input_ids,
                attention_mask = encoder_attention_mask,
                return_dict = True,
                output_attentions = True,
                output_hidden_states = True)
            self.encoder_outputs.append(encoder_outputs)
        self.encoder_attention_mask = encoder_attention_mask
        self.input_ids = input_ids
        self.lambdas = lambdas

    def __call__(self, input_ids, logits):

        log_logits_list = []
        # logits_list = []
        # avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(self.n_models)

        # we get the raw, unprocessed logits from the lead model
        # we will get a probability distribution and then multiply by the lambda (weight)
        log_logits_list.append(F.log_softmax(logits, dim=-1) + math.log(self.lambdas[0]))
        # logits_list.append(F.softmax(logits, dim=-1) * self.lambdas[0])

        # for each model, we get the logits and apply the softmax then multiply by the associated lambda (weight)
        for i, model in enumerate(self.ensembled_models):
            translation_model_logits = model.forward(
                decoder_input_ids = input_ids,
                encoder_outputs = self.encoder_outputs[i]).logits[:, -1, :]
            # logits_list.append(F.softmax(translation_model_logits, dim=-1) * self.lambdas[i+1])
            log_logits_list.append(F.log_softmax(translation_model_logits, dim=-1) + math.log(self.lambdas[i+1]))
        
        avg_probs = torch.logsumexp(torch.stack(log_logits_list, dim=0), dim=0)
        # softmax_logits = torch.sum(torch.stack(logits_list, dim=0), dim=0)


        # the resulting logits should be an interpolated probability distribution, we return the log probabilities
        return avg_probs


def yield_doc(istream):
    for line in istream:
        yield line.strip()


def get_batches(docstream):
    batch = []
    for d in docstream:
        batch.append(d)
        yield batch
        batch = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", '-m', nargs='+', required=True, help="List of models to ensemble")
    parser.add_argument('--input', '-i', type=str, default=None)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--seed', '-s', type=int, default=14)
    parser.add_argument('--lambdas', '-l', type=float, nargs='+', default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    istream = open(args.input, 'r') if args.input is not None else sys.stdin
    ostream = open(args.output, 'w') if args.output is not None else sys.stdout

    docstream  = yield_doc(istream)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # all tokenizers must be the same anyway
    tokenizer = AutoTokenizer.from_pretrained(args.models[0])

    models = []
    # load each of the marian models and add to list
    for m in args.models:
        marian_model = MarianMTModel.from_pretrained(m)
        marian_model.eval()
        marian_model = marian_model.to(device)
        models.append(marian_model)

    # if weights were not passed, all weights are even
    if args.lambdas is None:
        args.lambdas = [1.0] * len(models)

    assert (len(args.lambdas) == len(models)), "Number of lambda values must match number of models"

    # normalize weights to sum to 1
    if args.lambdas is not None:
        lambdas = [l / sum(args.lambdas) for l in args.lambdas]
    else:
        lambdas = [1.0 / len(models)] * len(models)

    for batch in get_batches(docstream):

        # tokenize batch. padding and attention masks are currently irrelevant because batch_size=1
        tokenization = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = tokenization["input_ids"].to(device)
        attention_mask = tokenization["attention_mask"].to(device)

        # the lead model is the one we will call generate on.
        # all other models will be passed to the "Mixture" which will ensemble the logits during each step
        lead_model = models[0]
        outputs = lead_model.generate(input_ids, 
                                    logits_processor=[Mixture(models[1:], input_ids, attention_mask, lambdas=lambdas)])
        # for each sentence in the batch, we append the generated sentence to the translation context
        for i, output in enumerate(outputs):
            print(tokenizer.decode(output, skip_special_tokens=True))