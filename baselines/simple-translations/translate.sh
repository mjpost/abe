rm -r outputs

mkdir -p outputs/en-de/sentences
mkdir -p outputs/en-es/sentences
mkdir -p outputs/en-cs/sentences
mkdir -p outputs/en-ru/sentences
mkdir -p outputs/en-uk/sentences

mkdir -p outputs/en-de/targets
mkdir -p outputs/en-es/targets
mkdir -p outputs/en-cs/targets
mkdir -p outputs/en-ru/targets
mkdir -p outputs/en-uk/targets

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep1 > outputs/en-de/sentences/baseline_en-de_8k_ep1
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep2 > outputs/en-de/sentences/baseline_en-de_8k_ep2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep3 > outputs/en-de/sentences/baseline_en-de_8k_ep3
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep4 > outputs/en-de/sentences/baseline_en-de_8k_ep4
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep5 > outputs/en-de/sentences/baseline_en-de_8k_ep5
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep10 > outputs/en-de/sentences/baseline_en-de_8k_ep10
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep15 > outputs/en-de/sentences/baseline_en-de_8k_ep15
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep20 > outputs/en-de/sentences/baseline_en-de_8k_ep20
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep25 > outputs/en-de/sentences/baseline_en-de_8k_ep25

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep1 > outputs/en-de/sentences/baseline_en-de_16k_ep1
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep2 > outputs/en-de/sentences/baseline_en-de_16k_ep2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep3 > outputs/en-de/sentences/baseline_en-de_16k_ep3
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep4 > outputs/en-de/sentences/baseline_en-de_16k_ep4
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep5 > outputs/en-de/sentences/baseline_en-de_16k_ep5
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep10 > outputs/en-de/sentences/baseline_en-de_16k_ep10
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep15 > outputs/en-de/sentences/baseline_en-de_16k_ep15
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep20 > outputs/en-de/sentences/baseline_en-de_16k_ep20
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep25 > outputs/en-de/sentences/baseline_en-de_16k_ep25

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep1 > outputs/en-de/sentences/baseline_en-de_32k_ep1
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep2 > outputs/en-de/sentences/baseline_en-de_32k_ep2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep3 > outputs/en-de/sentences/baseline_en-de_32k_ep3
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep4 > outputs/en-de/sentences/baseline_en-de_32k_ep4
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep5 > outputs/en-de/sentences/baseline_en-de_32k_ep5
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep10 > outputs/en-de/sentences/baseline_en-de_32k_ep10
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep15 > outputs/en-de/sentences/baseline_en-de_32k_ep15
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep20 > outputs/en-de/sentences/baseline_en-de_32k_ep20
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep25 > outputs/en-de/sentences/baseline_en-de_32k_ep25

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep1 > outputs/en-de/sentences/baseline_en-de_64k_ep1
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep2 > outputs/en-de/sentences/baseline_en-de_64k_ep2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep3 > outputs/en-de/sentences/baseline_en-de_64k_ep3
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep4 > outputs/en-de/sentences/baseline_en-de_64k_ep4
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep5 > outputs/en-de/sentences/baseline_en-de_64k_ep5
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep10 > outputs/en-de/sentences/baseline_en-de_64k_ep10
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep15 > outputs/en-de/sentences/baseline_en-de_64k_ep15
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep20 > outputs/en-de/sentences/baseline_en-de_64k_ep20
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep25 > outputs/en-de/sentences/baseline_en-de_64k_ep25


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-600M deu_Latn > outputs/en-de/sentences/nllb-200-distilled-600M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-1.3B deu_Latn > outputs/en-de/sentences/nllb-200-distilled-1.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-3.3B deu_Latn > outputs/en-de/sentences/nllb-200-3.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-1.3B deu_Latn > outputs/en-de/sentences/nllb-200-1.3B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_418M de > outputs/en-de/sentences/m2m100_418M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_1.2B de > outputs/en-de/sentences/m2m100_1.2B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-7B-v0.2 German > outputs/en-de/sentences/TowerInstruct-7B-v0.2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-Mistral-7B-v0.2 German > outputs/en-de/sentences/TowerInstruct-Mistral-7B-v0.2


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-1B-Instruct de > outputs/en-de/sentences/Llama-3.2-1B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-3B-Instruct de > outputs/en-de/sentences/Llama-3.2-3B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.1-8B-Instruct de > outputs/en-de/sentences/Llama-3.1-8B-Instruct-0-SHOT

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-1B-Instruct de > outputs/en-de/sentences/Llama-3.2-1B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-3B-Instruct de > outputs/en-de/sentences/Llama-3.2-3B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.1-8B-Instruct de > outputs/en-de/sentences/Llama-3.1-8B-Instruct-3-SHOT


############################################################################################
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-600M spa_Latn > outputs/en-es/sentences/nllb-200-distilled-600M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-1.3B spa_Latn > outputs/en-es/sentences/nllb-200-distilled-1.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-3.3B spa_Latn > outputs/en-es/sentences/nllb-200-3.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-1.3B spa_Latn > outputs/en-es/sentences/nllb-200-1.3B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_418M es > outputs/en-es/sentences/m2m100_418M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_1.2B es > outputs/en-es/sentences/m2m100_1.2B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-7B-v0.2 Spanish > outputs/en-es/sentences/TowerInstruct-7B-v0.2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-Mistral-7B-v0.2 Spanish > outputs/en-es/sentences/TowerInstruct-Mistral-7B-v0.2


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-1B-Instruct es > outputs/en-es/sentences/Llama-3.2-1B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-3B-Instruct es > outputs/en-es/sentences/Llama-3.2-3B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.1-8B-Instruct es > outputs/en-es/sentences/Llama-3.1-8B-Instruct-0-SHOT

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-1B-Instruct es > outputs/en-es/sentences/Llama-3.2-1B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-3B-Instruct es > outputs/en-es/sentences/Llama-3.2-3B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.1-8B-Instruct es > outputs/en-es/sentences/Llama-3.1-8B-Instruct-3-SHOT

############################################################################################
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-600M ces_Latn > outputs/en-cs/sentences/nllb-200-distilled-600M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-1.3B ces_Latn > outputs/en-cs/sentences/nllb-200-distilled-1.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-3.3B ces_Latn > outputs/en-cs/sentences/nllb-200-3.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-1.3B ces_Latn > outputs/en-cs/sentences/nllb-200-1.3B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_418M cs > outputs/en-cs/sentences/m2m100_418M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_1.2B cs > outputs/en-cs/sentences/m2m100_1.2B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-7B-v0.2 Czech > outputs/en-cs/sentences/TowerInstruct-7B-v0.2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-Mistral-7B-v0.2 Czech > outputs/en-cs/sentences/TowerInstruct-Mistral-7B-v0.2


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-1B-Instruct cs > outputs/en-cs/sentences/Llama-3.2-1B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-3B-Instruct cs > outputs/en-cs/sentences/Llama-3.2-3B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.1-8B-Instruct cs > outputs/en-cs/sentences/Llama-3.1-8B-Instruct-0-SHOT

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-1B-Instruct cs > outputs/en-cs/sentences/Llama-3.2-1B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-3B-Instruct cs > outputs/en-cs/sentences/Llama-3.2-3B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.1-8B-Instruct cs > outputs/en-cs/sentences/Llama-3.1-8B-Instruct-3-SHOT

############################################################################################
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-600M rus_Cyrl > outputs/en-ru/sentences/nllb-200-distilled-600M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-1.3B rus_Cyrl > outputs/en-ru/sentences/nllb-200-distilled-1.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-3.3B rus_Cyrl > outputs/en-ru/sentences/nllb-200-3.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-1.3B rus_Cyrl > outputs/en-ru/sentences/nllb-200-1.3B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_418M ru > outputs/en-ru/sentences/m2m100_418M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_1.2B ru > outputs/en-ru/sentences/m2m100_1.2B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-7B-v0.2 Russian > outputs/en-ru/sentences/TowerInstruct-7B-v0.2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-Mistral-7B-v0.2 Russian > outputs/en-ru/sentences/TowerInstruct-Mistral-7B-v0.2


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-1B-Instruct ru > outputs/en-ru/sentences/Llama-3.2-1B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-3B-Instruct ru > outputs/en-ru/sentences/Llama-3.2-3B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.1-8B-Instruct ru > outputs/en-ru/sentences/Llama-3.1-8B-Instruct-0-SHOT

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-1B-Instruct ru > outputs/en-ru/sentences/Llama-3.2-1B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-3B-Instruct ru > outputs/en-ru/sentences/Llama-3.2-3B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.1-8B-Instruct ru > outputs/en-ru/sentences/Llama-3.1-8B-Instruct-3-SHOT

############################################################################################
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-600M ukr_Cyrl > outputs/en-uk/sentences/nllb-200-distilled-600M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-1.3B ukr_Cyrl > outputs/en-uk/sentences/nllb-200-distilled-1.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-3.3B ukr_Cyrl > outputs/en-uk/sentences/nllb-200-3.3B
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-1.3B ukr_Cyrl > outputs/en-uk/sentences/nllb-200-1.3B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_418M uk > outputs/en-uk/sentences/m2m100_418M
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_1.2B uk > outputs/en-uk/sentences/m2m100_1.2B


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-7B-v0.2 Ukrainian > outputs/en-uk/sentences/TowerInstruct-7B-v0.2
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-Mistral-7B-v0.2 Ukrainian > outputs/en-uk/sentences/TowerInstruct-Mistral-7B-v0.2


cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-1B-Instruct uk > outputs/en-uk/sentences/Llama-3.2-1B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-3B-Instruct uk > outputs/en-uk/sentences/Llama-3.2-3B-Instruct-0-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.1-8B-Instruct uk > outputs/en-uk/sentences/Llama-3.1-8B-Instruct-0-SHOT

cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-1B-Instruct uk > outputs/en-uk/sentences/Llama-3.2-1B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-3B-Instruct uk > outputs/en-uk/sentences/Llama-3.2-3B-Instruct-3-SHOT
cat ../../refs/wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.1-8B-Instruct uk > outputs/en-uk/sentences/Llama-3.1-8B-Instruct-3-SHOT