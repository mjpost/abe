# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys, json
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict

model_one_id = sys.argv[1] # must be m2m
model_two_id = sys.argv[2] # must be m2m

device = "cuda" if torch.cuda.is_available() else "cpu"

# must be the m2m model
tokenizer_one = AutoTokenizer.from_pretrained(model_one_id)
tokenizer_one.src_lang = "en"
model_one = AutoModelForSeq2SeqLM.from_pretrained(model_one_id).to(device)
german_one = tokenizer_one.get_lang_id("de")

# must be the nllb model
tokenizer_two = AutoTokenizer.from_pretrained(model_two_id)
model_two = AutoModelForSeq2SeqLM.from_pretrained(model_two_id).to(device)
german_two = tokenizer_two.convert_tokens_to_ids("deu_Latn")

def get_model_score(model, tokenizer, inputs, output_one, output_two, ensemble_output, lang_token):

    def get_score(inputs, outputs, lang_token):
        input_ids = tokenizer.encode(inputs, return_tensors="pt", max_length=250, truncation=True).to(device)
        output_ids = tokenizer.encode(outputs, return_tensors="pt", max_length=250, truncation=True).to(device)
        output_ids[0][0] = lang_token
        # add the </s> token to the beggining
        torch.cat((
            torch.tensor([2], device=device),
            output_ids[0]
        )).view(1, -1)

        all_logits = []
        for i in range(1, output_ids.shape[-1]-1):
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
        'model_one': get_score(inputs, output_one, lang_token),
        'model_two': get_score(inputs, output_two, lang_token),
        'ensemble': get_score(inputs, ensemble_output, lang_token),
    }
    return output

def determine_preference(likelihood):

    sorted_preferences = sorted(likelihood.items(), key=lambda x: x[1], reverse=True)
    if sorted_preferences[0][1] == sorted_preferences[1][1]:
        if sorted_preferences[0][1] == sorted_preferences[2][1]:
            return set(['model_one', 'model_two', 'ensemble'])
        return set([sorted_preferences[0][0], sorted_preferences[1][0]])
    return set([sorted_preferences[0][0]])



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

ranking_agreement = {
    "model_one+model_two": 0,
    "model_one+ensemble": 0,
    "model_two+ensemble": 0,
}

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
    model_one_likelihood = get_model_score(model_one, tokenizer_one, model_inputs, model_one_output, model_two_output, ensemble_output, german_one)
    model_two_likelihood = get_model_score(model_two, tokenizer_two, model_inputs, model_one_output, model_two_output, ensemble_output, german_two)
    # except:
    #     print(line)
    #     exit()
    ensemble_likelihood = {
        'model_one': (model_one_likelihood['model_one'] + model_two_likelihood['model_one']) / 2,
        'model_two': (model_one_likelihood['model_two'] + model_two_likelihood['model_two']) / 2,
        'ensemble': (model_one_likelihood['ensemble'] + model_two_likelihood['ensemble']) / 2
    
    }

    model_one_preferences = determine_preference(model_one_likelihood)
    model_two_preferences = determine_preference(model_two_likelihood)
    ensemble_preferences = determine_preference(ensemble_likelihood)

    if len(model_one_preferences.intersection(model_two_preferences)) > 0:
        ranking_agreement['model_one+model_two'] += 1
    if len(model_one_preferences.intersection(ensemble_preferences)) > 0:
        ranking_agreement['model_one+ensemble'] += 1
    if len(model_two_preferences.intersection(ensemble_preferences)) > 0:
        ranking_agreement['model_two+ensemble'] += 1


    for preference in model_one_preferences:
        scores['model_one'][preference] += 1
    for preference in model_two_preferences:
        scores['model_two'][preference] += 1
    for preference in ensemble_preferences:
        scores['ensemble'][preference] += 1

print(json.dumps(scores, indent=2))
print(json.dumps(ranking_agreement, indent=2))
    