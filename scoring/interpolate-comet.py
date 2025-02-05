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


CUSTOM = []
for v in [8, 16, 32, 64]:
    for e in [1,2,3,4,5,10,15,20,25]:
        CUSTOM.append(f"baseline_en-de_{v}k_ep{e}")

MAJOR_CUSTOM = []
for v in [8, 16, 32, 64]:
    for e in [25]:
        MAJOR_CUSTOM.append(f"baseline_en-de_{v}k_ep{e}")

experiments = []
for i in range(len(CUSTOM)):
    for j in range(i+1, len(CUSTOM)):
        experiments.append((CUSTOM[i], CUSTOM[j]))


for exp in experiments:
    model_one = exp[0].replace('/', '-')
    model_two = exp[1].replace('/', '-')
    path = os.path.join('../baselines/interpolation/outputs/targets', f"{model_one}+{model_two}")
    if not os.path.exists(path):
        print(f"{model_one}\t{model_two}\t-")
    if os.path.exists(path):
        comet_score = get_comet_score(path)
        print(f"{model_one}\t{model_two}\t{comet_score:.1f}")
