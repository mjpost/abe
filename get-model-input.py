import os, sys

model_name = sys.argv[1]
language_pair = sys.argv[2]
testset = sys.argv[3]

TOWER_LIST = ["Unbabel/TowerInstruct-7B-v0.2", "Unbabel/TowerInstruct-Mistral-7B-v0.2"]
LLAMA_LIST = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
NLLB_LIST = ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-3.3B", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-1.3B"]
M2M_LIST = ["facebook/m2m100_418M", "facebook/m2m100_1.2B"]

source_lang = language_pair.split('-')[0]

if model_name in TOWER_LIST:
    print(f"input_data/{testset}.{language_pair}.{source_lang}.tower.json")
elif model_name in LLAMA_LIST:
    print(f"input_data/{testset}.{language_pair}.{source_lang}.llama3.json")
elif model_name in NLLB_LIST:
    print(f"input_data/{testset}.{language_pair}.{source_lang}.nllb.json")
elif model_name in M2M_LIST:
    print(f"input_data/{testset}.{language_pair}.{source_lang}.m2m.json")
else: # my models
    print(f"input_data/{testset}.{language_pair}.{source_lang}.bilingual-mt.json")
