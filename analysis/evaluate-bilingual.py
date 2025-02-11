# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys, json
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_one_id = sys.argv[1]
model_two_id = sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_one = AutoTokenizer.from_pretrained(model_one_id, torch_dtype=torch.bfloat16)
model_one = AutoModelForSeq2SeqLM.from_pretrained(model_one_id).to(device)

tokenizer_two = AutoTokenizer.from_pretrained(model_two_id, torch_dtype=torch.bfloat16)
model_two = AutoModelForSeq2SeqLM.from_pretrained(model_two_id).to(device)

def get_model_score(model, tokenizer, inputs, output_one, output_two, ensemble_output):

    def get_score(inputs, outputs):
        input_ids = tokenizer.encode(inputs, return_tensors="pt").to(device)
        output_ids = tokenizer.encode(outputs, return_tensors="pt").to(device)
        all_logits = []
        for i in range(output_ids.shape[-1]-1):
            outputs = model(
                input_ids = input_ids,
                decoder_input_ids = output_ids,
                return_dict = True,
                output_hidden_states=True
            )
            logits = outputs.logits[0]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)[-1]
            all_logits.append(logits[output_ids[0, i+1]].item())

        return sum(all_logits) / len(all_logits)
    output = {
        'model_one': get_score(inputs, output_one),
        'model_two': get_score(inputs, output_two),
        'ensemble': get_score(inputs, ensemble_output),
    }
    return output


scores = {
    "model_one": {
        "model_one": 0,
        "model_two": 0,
        "ensemble": 0
    },
    "model_two": {
        "model_one": 0,
        "model_two": 0,
        "ensemble": 0
    }
}
for line in sys.stdin:
    line = line.strip().split('\t')
    assert len(line) == 4
    model_inputs = line[0]
    model_one_output = line[1]
    model_two_output = line[2]
    ensemble_output = line[3]

    # print('Model inputs')
    # print(model_inputs)
    # print('Model one output')
    # print(model_one_output)
    # print('Model two output')
    # print(model_two_output)
    # print('Ensemble output')
    # print(ensemble_output)

    model_one_likelihood = get_model_score(model_one, tokenizer_one, model_inputs, model_one_output, model_two_output, ensemble_output)
    # print('Model one preferences')
    # print(json.dumps(model_one_likelihood, indent=2))
    if model_one_likelihood['model_one'] > model_one_likelihood['model_two'] and model_one_likelihood['model_one'] > model_one_likelihood['ensemble']:
        scores['model_one']['model_one'] += 1
    elif model_one_likelihood['model_two'] > model_one_likelihood['model_one'] and model_one_likelihood['model_two'] > model_one_likelihood['ensemble']:
        scores['model_one']['model_two'] += 1
    else:
        scores['model_one']['ensemble'] += 1

    model_two_likelihood = get_model_score(model_two, tokenizer_two, model_inputs, model_one_output, model_two_output, ensemble_output)
    # print('Model two preferences')
    # print(json.dumps(model_two_likelihood, indent=2))
    if model_two_likelihood['model_one'] > model_two_likelihood['model_two'] and model_two_likelihood['model_one'] > model_two_likelihood['ensemble']:
        scores['model_two']['model_one'] += 1
    elif model_two_likelihood['model_two'] > model_two_likelihood['model_one'] and model_two_likelihood['model_two'] > model_two_likelihood['ensemble']:
        scores['model_two']['model_two'] += 1
    else:
        scores['model_two']['ensemble'] += 1

print(json.dumps(scores, indent=2))
    