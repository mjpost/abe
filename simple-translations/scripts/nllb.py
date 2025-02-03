# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"


# only need to change the below if it's not English to start
tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)


for line in sys.stdin:
    line = line.strip()
    inputs = tokenizer(line, return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("deu_Latn"), max_length=256,
        num_beams = 5,
    )
    print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])