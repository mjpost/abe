import json
from collections import namedtuple
import pytest
import torch

from ensembling.models import get_models
from ensembling.ensemble import ensemble_beam_search

from ensembling.solo_scores import get_scores

EPSILON = 1e-4

MODEL_LIST = ["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"]
INPUTS = [
    [{"encoder_bos_tokens": ["eng_Latn"], "decoder_bos_tokens": ["fra_Latn"], "encoder_inputs": "This is a test."}],
    [{"encoder_bos_tokens": ["__en__"], "decoder_bos_tokens": ["__fr__"], "encoder_inputs": "This is a test."}]
]

test_inputs = [
    (MODEL_LIST, INPUTS)
]

@pytest.mark.parametrize("model_list, inputs", test_inputs)
def test_ensemble_scores(model_list, inputs):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    models = get_models(model_list, device=device, cache=False)

    weights = [1 / len(models) for _ in range(len(models))]

    outputs = ensemble_beam_search(
                inputs,
                models,
                weights = weights,
                max_length=128,
                num_beams=1)[0]
    
    input_ids = outputs['input_ids']
    output_ids = outputs['token_ids']
    token_scores = outputs['token_scores']

    for model, input_id, output_id, scores in zip(models, input_ids, output_ids, token_scores):
        individual_scores = get_scores({"input_ids": input_id, "output_ids": output_id}, model.model, model.bad_words, device=device)
        truncated_individual_scores = individual_scores[len(individual_scores)-len(scores):]
        for i, (a, b) in enumerate(zip(truncated_individual_scores, scores)):
            assert abs(a - b) < EPSILON

test_file_inputs = [
    (MODEL_LIST, "test/flores.short.dev.en-fr.tsv")
]

@pytest.mark.parametrize("model_list, file_path", test_file_inputs)
def test_ensemble_scores_file(model_list, file_path):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    models = get_models(model_list, device=device, cache=False)

    weights = [1 / len(models) for _ in range(len(models))]
    
    with open(file_path) as infile:
        for line in infile:
            inputs = [[json.loads(_)] for _ in line.split('\t')]
            assert len(inputs) == len(model_list), "Number of inputs does not match number of models"

            outputs = ensemble_beam_search(
                inputs,
                models,
                weights = weights,
                max_length=128,
                num_beams=1)[0]

            input_ids = outputs['input_ids']
            output_ids = outputs['token_ids']
            token_scores = outputs['token_scores']

            for model, input_id, output_id, scores in zip(models, input_ids, output_ids, token_scores):
                individual_scores = get_scores({"input_ids": input_id, "output_ids": output_id}, model.model, model.bad_words, device=device)
                truncated_individual_scores = individual_scores[len(individual_scores)-len(scores):]
                for i, (a, b) in enumerate(zip(truncated_individual_scores, scores)):
                    print(a, b, abs(a-b))
                    assert abs(a - b) < EPSILON