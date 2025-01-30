# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys
import torch
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

for line in sys.stdin:
    line = line.strip()
    messages = [
        {"role": "user", "content": f"Translate the following segment into German. English: {line}\nGerman:"}    
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1]['content'].replace('\n', ' ').strip())