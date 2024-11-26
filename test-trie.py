from ensembling.utils import Trie, Node



vocabulary = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" \
                , "he", "ll", "lo", "hello", "here", "help", "helsinki", "held"]

trie = Trie(len(vocabulary))

for i, token in enumerate(vocabulary):
    trie.add_string(token, i)

print("FOR H")
mask = trie.search_key("h")

for item_id, item in enumerate(mask):
    if item == 1:
        print(vocabulary[item_id], item_id)
print("\n\n")

print("FOR HE")
mask = trie.search_key("he")

for item_id, item in enumerate(mask):
    if item == 1:
        print(vocabulary[item_id], item_id)
print("\n\n")


print("FOR HEL")
mask = trie.search_key("hel")

for item_id, item in enumerate(mask):
    if item == 1:
        print(vocabulary[item_id], item_id)
print("\n\n")

print("FOR HELL")
mask = trie.search_key("hell")

for item_id, item in enumerate(mask):
    if item == 1:
        print(vocabulary[item_id], item_id)
print("\n\n")

print("FOR HELLO")
mask = trie.search_key("hello")

for item_id, item in enumerate(mask):
    if item == 1:
        print(vocabulary[item_id], item_id)
print("\n\n")

print("FOR LO")
mask = trie.search_key("lo")

for item_id, item in enumerate(mask):
    if item == 1:
        print(vocabulary[item_id], item_id)
print("\n\n")


#########################################################################

import torch
from transformers import AutoTokenizer
from ensembling.models import get_models, build_tries

test_strings = [" att", "tok", " tok", "hello", " hello"]

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocabulary = [tok for tok, _ in sorted(tokenizer.get_vocab().items(), key = lambda kv: kv[1])]
model = get_models([model_name], torch.device('cpu'), False, False)
build_tries(model)

print("Testing with model", model_name)
for test in test_strings:
    print(f"FOR '{test}'")
    mask = model[0].trie.search_key(test)

    for item_id, item in enumerate(mask):
        if item == 1:
            print(vocabulary[item_id], item_id)
    print("\n\n")


#########################################################################

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocabulary = [tok for tok, _ in sorted(tokenizer.get_vocab().items(), key = lambda kv: kv[1])]
model = get_models([model_name], torch.device('cpu'), False, False)
build_tries(model)

print("Testing with model", model_name)
for test in test_strings:
    print(f"FOR '{test}'")
    mask = model[0].trie.search_key(test)

    for item_id, item in enumerate(mask):
        if item == 1:
            print(vocabulary[item_id], item_id)
    print("\n\n")

#########################################################################

model_name = "rewicks/baseline_en-de_64k_ep25"
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocabulary = [tok for tok, _ in sorted(tokenizer.get_vocab().items(), key = lambda kv: kv[1])]
model = get_models([model_name], torch.device('cpu'), False, False)
build_tries(model)

mask = model[0].trie.search_key_indices("hello")

print("Testing with model", model_name)
for test in test_strings:
    print(f"FOR '{test}'")
    mask = model[0].trie.search_key(test)

    for item_id, item in enumerate(mask):
        if item == 1:
            print(vocabulary[item_id], item_id)
    print("\n\n")

