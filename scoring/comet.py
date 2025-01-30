import os
import sacrebleu
from pymarian import Evaluator
from subprocess import check_output


MARIAN_CLI = "- -m wmt22-comet-da -a only -d 0"
#MARIAN_CLI = "--help"
#COMET_EVALUATDDOR = Evaluator(MARIAN_CLI)

SOURCE_PATH="/exp/rwicks/ensemble24/refs/wmt24.en-de.en"
SOURCES = [open(SOURCE_PATH).readlines()]

REFERENCE_PATH="/exp/rwicks/ensemble24/refs/wmt24.en-de.de"
REFERENCES = [open(REFERENCE_PATH).readlines()]

"""
def get_comet_score(file_path):
    data = []
    with open(file_path) as hypotheses:
        for hyp, src, ref in zip(hypotheses, SOURCES, REFERENCES):
            data.append([src, hyp, ref])
    breakpoint()
    scores = COMET_EVALUATOR.run(data)   
"""

def get_comet_score(file_path):
    out = check_output(f"paste {SOURCE_PATH} {file_path} {REFERENCE_PATH} | pymarian-eval {MARIAN_CLI}", shell=True).decode('utf-8').strip()
    return float(out)*100


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

experiments = []
# baseline_en-de_8k_ep25+Llama-3.1-8B-Instruct-0-SHOT
# bilingual x bilingual
for i in range(len(RACHELS)):
    for j in range(i+1, len(RACHELS)):
        experiments.append((RACHELS[i], RACHELS[j]))

# bilingual x m2m
for rachel in MAJOR_RACHELS:
    for m2m in M2M_LIST:
        experiments.append((rachel, m2m))

# bilingual x nllb
for rachel in MAJOR_RACHELS:
    for nllb in NLLB_LIST:
        experiments.append((rachel, nllb))

# bilingual x tower
for rachel in MAJOR_RACHELS:
    for tower in TOWER_LIST:
        experiments.append((rachel, tower))

# bilingual x llama
for rachel in MAJOR_RACHELS:
    for llama in LLAMA_LIST:
        experiments.append((rachel, llama))

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

import os
import sacrebleu

REFERENCE_PATH = "/exp/rwicks/ensemble24/refs/wmt24.en-de.de"
REFERENCES = [open(REFERENCE_PATH).readlines()]

for exp in experiments:
    model_one = '-'.join(exp[0].split('/')[1:])
    model_two = '-'.join(exp[1].split('/')[1:])
    path = os.path.join('/exp/rwicks/ensemble24/translations/wmt24/en-de/targets/', f"{model_one}+{model_two}")
    # print(path)
    if not os.path.exists(path):
        print(f"{model_one}\t{model_two}\t-")
    if os.path.exists(path):
        bleu_score = get_comet_score(path)
        print(f"{model_one}\t{model_two}\t{bleu_score:.1f}")
