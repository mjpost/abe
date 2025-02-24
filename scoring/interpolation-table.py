from collections import defaultdict
import numpy as np
import sys

vocab=sys.argv[1]

lang="de"
def read_scores(input_path, columns=3):
    scores = defaultdict(dict) if columns == 3 else {}
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[-1] != '-':
                if columns == 3:
                    if not parts[0].startswith('baseline') or not parts[1].startswith('baseline'):
                        continue
                    if vocab not in parts[0] or vocab not in parts[1]:
                        continue
                    scores[parts[0]][parts[1]] = float(parts[2])
                    scores[parts[1]][parts[0]] = float(parts[2])
                else:
                    if not parts[0].startswith('baseline'):
                        continue
                    if vocab not in parts[0]:
                        continue
                    scores[parts[0]] = float(parts[1])
    return scores

BLEU_SCORES_PATH = f'scores/{lang}/bleu'
SIMPLE_BLEU_SCORES_PATH = f'scores/{lang}/simple-bleu'
BLEU_INTERPOLATE_SCORES_PATH = f'scores/{lang}/interpolate-bleu'

COMET_SCORES_PATH = f'scores/{lang}/comet'
COMET_SIMPLE_SCORES_PATH = f'scores/{lang}/simple-comet'
COMET_INTERPOLATE_SCORES_PATH = f'scores/{lang}/interpolate-comet'

bleu_scores = read_scores(BLEU_SCORES_PATH)
simple_bleu_scores = read_scores(SIMPLE_BLEU_SCORES_PATH, columns=2)
interpolate_bleu_scores = read_scores(BLEU_INTERPOLATE_SCORES_PATH)

comet_scores = read_scores(COMET_SCORES_PATH)
simple_comet_scores = read_scores(COMET_SIMPLE_SCORES_PATH, columns=2)
interpolate_comet_scores = read_scores(COMET_INTERPOLATE_SCORES_PATH)

average_simple_bleu_scores = sum(simple_bleu_scores.values()) / len(simple_bleu_scores)
print(simple_bleu_scores.keys())
# 1110.2999999999997
# len() = 40
# 37.75749999999999
print(sum(simple_bleu_scores.values()))
print(len(simple_bleu_scores))
print(average_simple_bleu_scores)

average_simple_comet_scores = sum(simple_comet_scores.values()) / len(simple_comet_scores)

def get_average_baseline_improvement(scores, simple_scores):
    values = []
    visited = set()
    for model_one in scores.keys():
        vocab_one = model_one.split('_')[2]
        for model_two in scores[model_one].keys():
            vocab_two = model_two.split('_')[2]
            if vocab_one != vocab_two:
                continue
            if f'{model_one}-{model_two}' not in visited and f'{model_two}-{model_one}' not in visited:
                delta = scores[model_one][model_two] - max(simple_scores.get(model_one, 0), simple_scores.get(model_two, 0))
                print(f'{model_one}-{model_two}\t{delta}')
                values.append(delta)
                visited.add(f'{model_one}-{model_two}')
                visited.add(f'{model_two}-{model_one}')
    return values


def get_average_interpolation_improvement(interpolate_scores, simple_scores):
    values = []
    visited = set()
    for model_one in interpolate_scores.keys():
        vocab_one = model_one.split('_')[2]
        for model_two in interpolate_scores[model_one].keys():
            vocab_two = model_two.split('_')[2]
            if vocab_one != vocab_two:
                continue
            if f'{model_one}-{model_two}' not in visited and f'{model_two}-{model_one}' not in visited:
                delta = interpolate_scores[model_one][model_two] - max(simple_scores.get(model_one, 0), simple_scores.get(model_two, 0))
                print(f'{model_one}-{model_two}\t{delta}')
                values.append(delta)
                visited.add(f'{model_one}-{model_two}')
                visited.add(f'{model_two}-{model_one}')
    return values


print("ABE IMPROVMENTS")
print("---------------------------------------------------------")
abe_improvement = get_average_baseline_improvement(bleu_scores, simple_bleu_scores)

print("\n\nINTERPOLATION IMPROVMENTS")
print("---------------------------------------------------------")
interpolation_improvement = get_average_interpolation_improvement(interpolate_bleu_scores, simple_bleu_scores)

print(interpolation_improvement)

print(f"Average Simple BLEU: {average_simple_bleu_scores}")
print(f"Average ABE: {np.average(abe_improvement)}")
print(f"Average Interpolation Improvement: {np.average(interpolation_improvement)}")

# abe_improvement = get_average_baseline_improvement(comet_scores, simple_comet_scores)
# interpolation_improvement = get_average_interpolation_improvement(interpolate_comet_scores, simple_comet_scores)

# print(interpolation_improvement)

# print(f"Average Simple COMET: {average_simple_comet_scores}")
# print(f"Average ABE: {np.average(abe_improvement)}")
# print(f"Average Interpolation Improvement: {np.average(interpolation_improvement)}")
