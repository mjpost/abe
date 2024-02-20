#!/usr/bin/env python3

import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

for line in sys.stdin:
    article = line.rstrip()
    inputs = tokenizer(article, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=30
    )

    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    print(translated_tokens)
