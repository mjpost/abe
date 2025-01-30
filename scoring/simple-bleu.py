
TOWER_LIST = ["Unbabel/TowerInstruct-7B-v0.2", "Unbabel/TowerInstruct-Mistral-7B-v0.2"]
LLAMA_LIST = ["meta-llama/Llama-3.2-1B-Instruct/3-SHOT", "meta-llama/Llama-3.2-1B-Instruct/0-SHOT", \
              "meta-llama/Llama-3.2-3B-Instruct/3-SHOT", "meta-llama/Llama-3.2-3B-Instruct/0-SHOT", \
                "meta-llama/Llama-3.1-8B-Instruct/3-SHOT", "meta-llama/Llama-3.1-8B-Instruct/0-SHOT"]
NLLB_LIST = ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-3.3B", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-1.3B"]
M2M_LIST = ["facebook/m2m100_418M", "facebook/m2m100_1.2B"]

RACHELS = []
for v in [8, 16, 32, 64]:
    for e in [1,2,3,4,5,10,15,20,25]:
        RACHELS.append(f"rewicks/baseline_en-de_{v}k_ep{e}")

MAJOR_RACHELS = []
for v in [8, 16, 32, 64]:
    for e in [25]:
        MAJOR_RACHELS.append(f"rewicks/baseline_en-de_{v}k_ep{e}")


import os
import sacrebleu

REFERENCE_PATH = "/exp/rwicks/ensemble24/refs/wmt24.en-de.de"
REFERENCES = [open(REFERENCE_PATH).readlines()]

def get_bleu_score(file_path):
    hypotheses = open(file_path).readlines()
    bleu = sacrebleu.corpus_bleu(hypotheses, REFERENCES).score
    return bleu

MODELS = RACHELS + MAJOR_RACHELS + TOWER_LIST + LLAMA_LIST + NLLB_LIST + M2M_LIST

for model in MODELS:
    model = '/'.join(model.split('/')[1:]).replace('/', '-')
    path = os.path.join('/exp/rwicks/ensemble24/simple-translations/outputs/targets/', model)
    if not os.path.exists(path):
        print(f"{model}\t-")
    if os.path.exists(path):
        bleu_score = get_bleu_score(path)
        print(f"{model}\t{bleu_score:.1f}")
