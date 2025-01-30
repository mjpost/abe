# https://huggingface.co/facebook/m2m100_418M
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import torch
import sys


model_id = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = M2M100ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
tokenizer = M2M100Tokenizer.from_pretrained(model_id)

tokenizer.src_lang = "en"

for line in sys.stdin:
    line = line.strip()
    inputs = tokenizer(line, return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("__de__"), max_length=256
    )
    print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])