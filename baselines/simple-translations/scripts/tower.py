# https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2

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
)

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

for line in sys.stdin:
    line = line.strip()
    messages = [
        {"role": "user", "content": f"Translate the following segment into {target}. English: {line}\n{target}:"}    
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        num_beams = 5,
        eos_token_id=terminators,
        do_sample = False,
    )
    print(outputs[0]['generated_text'].replace(prompt, "").replace('\n', ' ').strip())
    # print(outputs[0]["generated_text"][-1]['content'].replace('\n', ' ').strip())