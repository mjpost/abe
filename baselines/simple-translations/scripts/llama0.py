# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys
import torch
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = sys.argv[1]
target = sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=128001
)

prefixes = {
    "cs": "Translate the following text from English into Czech. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: ",
    "de": "Translate the following text from English into German. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: ",
    "es": "Translate the following text from English into Spanish. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: ",
    "uk": "Translate the following text from English into Ukrainian. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: ",
    "ru": "Translate the following text from English into Russian. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: "
}

for line in sys.stdin:
    line = line.strip()
    messages = [
        {"role": "user", "content": f"{prefixes[target]} {line}"}    
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
        num_beams = 5,
    )
    print(outputs[0]["generated_text"][-1]['content'].replace('\n', ' ').strip())