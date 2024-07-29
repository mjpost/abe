#!/usr/bin/env python3

import sys
from transformers import NllbTokenizer

tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")


for line in sys.stdin:
    line = line.rstrip()
    tok = tokenizer(line)

    print("LINE", line)
    print("TOKE", tok.input_ids)

# 2: '</s>'
# 256047 : 'eng_Latn'
