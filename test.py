import json
import torch
import sys

from ensembling.models import get_models, build_tries
from ensembling.ensemble import ensemble_beam_search, ensemble_sample

from ensembling.solo_scores import get_scores


MODEL_LIST = ["facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"]
INPUTS = [
    [{"encoder_bos_tokens": ["eng_Latn"], "decoder_bos_tokens": ["fra_Latn"], "encoder_inputs": "This is a test."}],
    [{"encoder_bos_tokens": ["__en__"], "decoder_bos_tokens": ["__fr__"], "encoder_inputs": "This is a test."}]
]
EPSILON = 1e-4

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
            print(a, b, abs(a-b))
            assert abs(a - b) < EPSILON

def test_ensemble_sample_scores(model_list, inputs):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    models = get_models(model_list, device=device, cache=False)
    build_tries(models)

    weights = [1 / len(models) for _ in range(len(models))]

    outputs = ensemble_sample(
                        inputs,
                        models,
                        weights,
                        num_samples = 1,
                        max_length = 128,
                        temperature = 1.0,
                        top_k = -1,
                        top_p = 1.0)[0][0]
    
    input_ids = outputs['input_ids']
    output_ids = outputs['token_ids']
    token_scores = outputs['token_scores']

    for model, input_id, output_id, scores in zip(models, input_ids, output_ids, token_scores):
        individual_scores = get_scores({"input_ids": input_id, "output_ids": output_id}, model.model, model.bad_words, device=device)
        truncated_individual_scores = individual_scores[len(individual_scores)-len(scores):]
        for i, (a, b) in enumerate(zip(truncated_individual_scores, scores)):
            print(a, b, abs(a-b))
            assert abs(a - b) < EPSILON


test_ensemble_scores(MODEL_LIST, INPUTS)
test_ensemble_sample_scores(MODEL_LIST, INPUTS)


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

def test_ensemble_sample_scores_files(model_list, file_path):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    models = get_models(model_list, device=device, cache=False)
    build_tries(models)

    weights = [1 / len(models) for _ in range(len(models))]

    with open(file_path) as infile:
        for line in infile:
            inputs = [[json.loads(_)] for _ in line.split('\t')]
            assert len(inputs) == len(model_list), "Number of inputs does not match number of models"

            outputs = ensemble_sample(
                                inputs,
                                models,
                                weights,
                                num_samples = 1,
                                max_length = 128,
                                temperature = 1.0,
                                top_k = -1,
                                top_p = 1.0)[0][0]
    
            input_ids = outputs['input_ids']
            output_ids = outputs['token_ids']
            token_scores = outputs['token_scores']

            for model, input_id, output_id, scores in zip(models, input_ids, output_ids, token_scores):
                individual_scores = get_scores({"input_ids": input_id, "output_ids": output_id}, model.model, model.bad_words, device=device)
                truncated_individual_scores = individual_scores[len(individual_scores)-len(scores):]
                for i, (a, b) in enumerate(zip(truncated_individual_scores, scores)):
                    print(a, b, abs(a-b))
                    assert abs(a - b) < EPSILON

test_ensemble_scores_file(MODEL_LIST, "test/flores.short.dev.en-fr.tsv")
test_ensemble_sample_scores_file(MODEL_LIST, "test/flores.short.dev.en-fr.tsv")