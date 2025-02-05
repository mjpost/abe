import os
from subprocess import check_output


MARIAN_CLI = "- -m wmt22-comet-da -a only -d 0"

SOURCE_PATH="../refs/wmt24.en-de.en"
SOURCES = [open(SOURCE_PATH).readlines()]

REFERENCE_PATH="../refs/wmt24.en-de.de"
REFERENCES = [open(REFERENCE_PATH).readlines()]


def get_comet_score(file_path):
    out = check_output(f"paste {SOURCE_PATH} {file_path} {REFERENCE_PATH} | pymarian-eval {MARIAN_CLI}", shell=True).decode('utf-8').strip()
    return float(out)*100


TOWER_LIST = ["TowerInstruct-7B-v0.2", "TowerInstruct-Mistral-7B-v0.2"]
LLAMA_LIST = ["Llama-3.2-1B-Instruct/3-SHOT", "Llama-3.2-1B-Instruct/0-SHOT", \
              "Llama-3.2-3B-Instruct/3-SHOT", "Llama-3.2-3B-Instruct/0-SHOT", \
                "Llama-3.1-8B-Instruct/3-SHOT", "Llama-3.1-8B-Instruct/0-SHOT"]
NLLB_LIST = ["nllb-200-distilled-600M", "nllb-200-3.3B", "nllb-200-distilled-1.3B", "nllb-200-1.3B"]
M2M_LIST = ["m2m100_418M", "m2m100_1.2B"]

CUSTOM = []
for v in [8, 16, 32, 64]:
    for e in [1,2,3,4,5,10,15,20,25]:
        CUSTOM.append(f"baseline_en-de_{v}k_ep{e}")

MAJOR_CUSTOM = []
for v in [8, 16, 32, 64]:
    for e in [25]:
        MAJOR_CUSTOM.append(f"baseline_en-de_{v}k_ep{e}")

MODELS = CUSTOM + MAJOR_CUSTOM + TOWER_LIST + LLAMA_LIST + NLLB_LIST + M2M_LIST

for model in MODELS:
    model = model.replace('/', '-')
    path = os.path.join('../baselines/simple-translations/outputs/targets/', model)
    if not os.path.exists(path):
        print(f"{model}\t-")
    if os.path.exists(path):
        bleu_score = get_comet_score(path)
        print(f"{model}\t{bleu_score:.1f}")
