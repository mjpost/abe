import sys
target=sys.argv[1]

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

import os
import sacrebleu

REFERENCE_PATH = f"../refs/wmt24.en-{target}.{target}"
REFERENCES = [open(REFERENCE_PATH).readlines()]

def get_bleu_score(file_path):
    hypotheses = open(file_path).readlines()
    bleu = sacrebleu.corpus_bleu(hypotheses, REFERENCES).score
    return bleu

for exp in experiments:
    model_one = exp[0].replace('/', '-')
    model_two = exp[1].replace('/', '-')
    path = os.path.join(f'../baselines/interpolation/outputs/en-{target}/targets', f"{model_one}+{model_two}")
    # print(path)
    if not os.path.exists(path):
        print(f"{model_one}\t{model_two}\t-")
    if os.path.exists(path):
        bleu_score = get_bleu_score(path)
        print(f"{model_one}\t{model_two}\t{bleu_score:.1f}")
