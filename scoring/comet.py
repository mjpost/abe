import sys
target=sys.argv[1]


import os
from subprocess import check_output


MARIAN_CLI = "- -m wmt22-comet-da -a only -d 0"

SOURCE_PATH=f"../refs/wmt24.en-{target}.en"
SOURCES = [open(SOURCE_PATH).readlines()]

REFERENCE_PATH=f"../refs/wmt24.en-{target}.{target}"
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

experiments = []
# bilingual x bilingual
for i in range(len(CUSTOM)):
    for j in range(i+1, len(CUSTOM)):
        experiments.append((CUSTOM[i], CUSTOM[j]))

# bilingual x m2m
for custom in MAJOR_CUSTOM:
    for m2m in M2M_LIST:
        experiments.append((custom, m2m))

# bilingual x nllb
for custom in MAJOR_CUSTOM:
    for nllb in NLLB_LIST:
        experiments.append((custom, nllb))

# bilingual x tower
for custom in MAJOR_CUSTOM:
    for tower in TOWER_LIST:
        experiments.append((custom, tower))

# bilingual x llama
for custom in MAJOR_CUSTOM:
    for llama in LLAMA_LIST:
        experiments.append((custom, llama))

# m2m x m2m
experiments.append((M2M_LIST[0], M2M_LIST[1]))

# m2m x nllb
for m2m in M2M_LIST:
    for nllb in NLLB_LIST:
        experiments.append((m2m, nllb))

# m2m x tower
for m2m in M2M_LIST:
    for tower in TOWER_LIST:
        experiments.append((m2m, tower))

# m2m x llama
for m2m in M2M_LIST:
    for llama in LLAMA_LIST:
        experiments.append((m2m, llama))

# nllb x nllb
for i in range(len(NLLB_LIST)):
    for j in range(i+1, len(NLLB_LIST)):
        experiments.append((NLLB_LIST[i], NLLB_LIST[j]))

# nllb x tower
for nllb in NLLB_LIST:
    for tower in TOWER_LIST:
        experiments.append((nllb, tower))

# nllb x llama
for nllb in NLLB_LIST:
    for llama in LLAMA_LIST:
        experiments.append((nllb, llama))

# tower x llama 
for tower in TOWER_LIST:
    for llama in LLAMA_LIST:
        experiments.append((tower, llama))

for exp in experiments:
    model_one = exp[0].replace('/', '-')
    model_two = exp[1].replace('/', '-')
    path = os.path.join(f'../translations/wmt24/en-{target}/targets/', f"{model_one}+{model_two}")
    if not os.path.exists(path):
        print(f"{model_one}\t{model_two}\t-")
    if os.path.exists(path):
        comet_score = get_comet_score(path)
        print(f"{model_one}\t{model_two}\t{comet_score:.1f}")
