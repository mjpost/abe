import json
import sys
from collections import Counter

from transformers import AutoTokenizer

model_one = sys.argv[1]
model_two = sys.argv[2]

tokenizer_one = AutoTokenizer.from_pretrained(model_one)
tokenizer_two = AutoTokenizer.from_pretrained(model_two)

counts = Counter()

for line in sys.stdin:
    line = json.loads(line.strip())
    sequence = line['sequence']
    tokens = line['tokens']
    tokens_one = tokens[0][1:-1] # ignore start and stop tokens
    tokens_two = tokens[1][1:-1] # ignore start and stop tokens

    model_one_tokenization = tokenizer_one.tokenize(sequence)
    model_two_tokenization = tokenizer_two.tokenize(sequence)

    for a, b in zip(tokens_one, model_one_tokenization):
        if a != b:
            print(tokens_one, model_one_tokenization)
            counts['model_one'] += 1
            break
    
    for a, b in zip(tokens_two, model_two_tokenization):
        if a != b:
            print(tokens_two, model_two_tokenization)
            counts['model_two'] += 1
            break

print(counts)
