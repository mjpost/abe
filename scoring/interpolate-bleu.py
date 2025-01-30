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

import os
import sacrebleu

REFERENCE_PATH = "/exp/rwicks/ensemble24/refs/wmt24.en-de.de"
REFERENCES = [open(REFERENCE_PATH).readlines()]

def get_bleu_score(file_path):
    hypotheses = open(file_path).readlines()
    bleu = sacrebleu.corpus_bleu(hypotheses, REFERENCES).score
    return bleu

for exp in experiments:
    model_one = '-'.join(exp[0].split('/')[1:])
    model_two = '-'.join(exp[1].split('/')[1:])
    path = os.path.join('/exp/rwicks/ensemble24/interpolation/outputs/targets', f"{model_one}+{model_two}")
    # print(path)
    if not os.path.exists(path):
        print(f"{model_one}\t{model_two}\t-")
    if os.path.exists(path):
        bleu_score = get_bleu_score(path)
        print(f"{model_one}\t{model_two}\t{bleu_score:.1f}")
