# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys, json
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict

model_one_id = sys.argv[1]
model_two_id = sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_one = AutoTokenizer.from_pretrained(model_one_id, torch_dtype=torch.bfloat16)
model_one = AutoModelForSeq2SeqLM.from_pretrained(model_one_id).to(device)

tokenizer_two = AutoTokenizer.from_pretrained(model_two_id, torch_dtype=torch.bfloat16)
model_two = AutoModelForSeq2SeqLM.from_pretrained(model_two_id).to(device)

def get_model_score(model, tokenizer, inputs, output_one, output_two, ensemble_output):

    def get_score(inputs, outputs):
        input_ids = tokenizer.encode(inputs, return_tensors="pt", max_length=255, truncation=True).to(device)
        output_ids = tokenizer.encode(outputs, return_tensors="pt", max_length=255, truncation=True).to(device)
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

def determine_preference(likelihood):

    sorted_preferences = sorted(likelihood.items(), key=lambda x: x[1], reverse=True)
    if sorted_preferences[0][1] == sorted_preferences[1][1]:
        if sorted_preferences[0][1] == sorted_preferences[2][1]:
            return ['model_one', 'model_two', 'ensemble']
        return [sorted_preferences[0][0], sorted_preferences[1][0]]
    return [sorted_preferences[0][0]]



scores = defaultdict(lambda: defaultdict(int))

# scores = {
#     "model_one": {
#         "model_one": 0,
#         "model_two": 0,
#         "ensemble": 0
#     },
#     "model_two": {
#         "model_one": 0,
#         "model_two": 0,
#         "ensemble": 0
#     },
#     "ensemble": {
#         "model_one": 0,
#         "model_two": 0,
#         "ensemble": 0,
#     }
# }
for line in sys.stdin:
    print
    line = line.split('\t')
    assert len(line) == 4, f'{line}'
    if len(line[0]) == 0 or len(line[1]) == 0 or len(line[2]) == 0 or len(line[3]) == 0:
        continue
    model_inputs = line[0]
    model_one_output = line[1]
    model_two_output = line[2]
    ensemble_output = line[3]


    # get scores (which determine preferences)
    # try:
    model_one_likelihood = get_model_score(model_one, tokenizer_one, model_inputs, model_one_output, model_two_output, ensemble_output)
    model_two_likelihood = get_model_score(model_two, tokenizer_two, model_inputs, model_one_output, model_two_output, ensemble_output)
    # except:
    #     print(line)
    #     exit()
    ensemble_likelihood = {
        'model_one': (model_one_likelihood['model_one'] + model_two_likelihood['model_one']) / 2,
        'model_two': (model_one_likelihood['model_two'] + model_two_likelihood['model_two']) / 2,
        'ensemble': (model_one_likelihood['ensemble'] + model_two_likelihood['ensemble']) / 2
    
    }

    for preference in determine_preference(model_one_likelihood):
        scores['model_one'][preference] += 1
    for preference in determine_preference(model_two_likelihood):
        scores['model_two'][preference] += 1
    for preference in determine_preference(ensemble_likelihood):
        scores['ensemble'][preference] += 1

print(json.dumps(scores, indent=2))
    