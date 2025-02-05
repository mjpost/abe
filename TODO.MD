# TODO

* [ ] qualify the difference in outputs between interpolation-ensembling and agreement-based-ensembling. Randomly selectin two model checkpoints and reviewing the differences
    * [ ] comet-score individual sentences, and sort on delta <-- qualify the differences

* [ ] try another language pair (European languages; Czech, Spanish, Russian, Ukrainian) (nllb x m2m; skip LLAMA 1B, 600M, skip smallest versions) LLama 8B 0-shot 3-shot
        * NLLB large x M2M large
        * NLLB large x Tower 7B - Instructv0.2 (skip Mistral)
        * NLLB large x LLama 8B
        * Tower 7B x LLama8B

* [ ] Double check the scoring scripts
    * find a random line in `bleu-scores`, score the accompanying file with `sacrebleu` cli. Is it the same?
    * find a random cell in analysis.ipnyb. Find the delta. Score the ensembled models, score the individual models. Is the delta correct?

# DOUBLE CHECKING OF RESULTS

Run everything on GPU, so we're not debating gpu vs cpu torch differences.

We have faith that our ensembled numbers cannot be inflated. Therefore, we need to check the baselines.

## M2M

* [ ] Check `simple-translations/scripts/m2m.py`. Does this look reasonable and follow the intended translation method at https://huggingface.co/facebook/m2m100_418M? Does it use beam=5?
* [ ] If you translate a random file, do you get the same output?

## NLLB

* [ ] Check `simple-translations/scripts/nllb.py` Does this look reasonable and follow the intended translation method at https://huggingface.co/docs/transformers/en/model_doc/nllb? Does it use beam=5?
* [ ] If you translate a random file, do you get the same output?

## LLAMA 

* [ ] Check `simple-translations/scripts/llama{0,3}.py`. Does this look reasonable and follow the intended generation method at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct? Does it use beam = 5?
* [ ] If you translate a random file, do you get the same output?


## TOWER

* [ ] Check `simple-translations/scripts/tower.py` Does this look reasonbale and follow the intended generation method at https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2? Does it use beam = 5?
* [ ] If you translate a random file, do you get the same output?


## Ensembling Double Check

* [ ] Random select a model pair and run `ensemble.py`. Do you get the same output?