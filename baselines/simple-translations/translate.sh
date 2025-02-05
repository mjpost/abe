cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-600M > outputs/sentences/nllb-200-distilled-600M
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-distilled-1.3B > outputs/sentences/nllb-200-distilled-1.3B
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-3.3B > outputs/sentences/nllb-200-3.3B
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/nllb.py facebook/nllb-200-1.3B > outputs/sentences/nllb-200-1.3B


cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_418M > outputs/sentences/m2m100_418M
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/m2m.py facebook/m2m100_1.2B > outputs/sentences/m2m100_1.2B


cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep1 > outputs/sentences/baseline_en-de_8k_ep1
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep2 > outputs/sentences/baseline_en-de_8k_ep2
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep3 > outputs/sentences/baseline_en-de_8k_ep3
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep4 > outputs/sentences/baseline_en-de_8k_ep4
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep5 > outputs/sentences/baseline_en-de_8k_ep5
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep10 > outputs/sentences/baseline_en-de_8k_ep10
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep15 > outputs/sentences/baseline_en-de_8k_ep15
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep20 > outputs/sentences/baseline_en-de_8k_ep20
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_8k_ep25 > outputs/sentences/baseline_en-de_8k_ep25

cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep1 > outputs/sentences/baseline_en-de_16k_ep1
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep2 > outputs/sentences/baseline_en-de_16k_ep2
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep3 > outputs/sentences/baseline_en-de_16k_ep3
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep4 > outputs/sentences/baseline_en-de_16k_ep4
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep5 > outputs/sentences/baseline_en-de_16k_ep5
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep10 > outputs/sentences/baseline_en-de_16k_ep10
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep15 > outputs/sentences/baseline_en-de_16k_ep15
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep20 > outputs/sentences/baseline_en-de_16k_ep20
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_16k_ep25 > outputs/sentences/baseline_en-de_16k_ep25

cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep1 > outputs/sentences/baseline_en-de_32k_ep1
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep2 > outputs/sentences/baseline_en-de_32k_ep2
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep3 > outputs/sentences/baseline_en-de_32k_ep3
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep4 > outputs/sentences/baseline_en-de_32k_ep4
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep5 > outputs/sentences/baseline_en-de_32k_ep5
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep10 > outputs/sentences/baseline_en-de_32k_ep10
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep15 > outputs/sentences/baseline_en-de_32k_ep15
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep20 > outputs/sentences/baseline_en-de_32k_ep20
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_32k_ep25 > outputs/sentences/baseline_en-de_32k_ep25

cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep1 > outputs/sentences/baseline_en-de_64k_ep1
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep2 > outputs/sentences/baseline_en-de_64k_ep2
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep3 > outputs/sentences/baseline_en-de_64k_ep3
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep4 > outputs/sentences/baseline_en-de_64k_ep4
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep5 > outputs/sentences/baseline_en-de_64k_ep5
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep10 > outputs/sentences/baseline_en-de_64k_ep10
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep15 > outputs/sentences/baseline_en-de_64k_ep15
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep20 > outputs/sentences/baseline_en-de_64k_ep20
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/bilingual.py rewicks/baseline_en-de_64k_ep25 > outputs/sentences/baseline_en-de_64k_ep25


cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-7B-v0.2 > outputs/sentences/TowerInstruct-7B-v0.2
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/tower.py Unbabel/TowerInstruct-Mistral-7B-v0.2 > outputs/sentences/TowerInstruct-Mistral-7B-v0.2


cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-1B-Instruct > outputs/sentences/Llama-3.2-1B-Instruct-0-SHOT
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.2-3B-Instruct > outputs/sentences/Llama-3.2-3B-Instruct-0-SHOT
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama0.py meta-llama/Llama-3.1-8B-Instruct > outputs/sentences/Llama-3.1-8B-Instruct-0-SHOT

cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-1B-Instruct > outputs/sentences/Llama-3.2-1B-Instruct-3-SHOT
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.2-3B-Instruct > outputs/sentences/Llama-3.2-3B-Instruct-3-SHOT
cat ../wmt24.en-de.en.sentences | cut -f2 | python -u scripts/llama3.py meta-llama/Llama-3.1-8B-Instruct > outputs/sentences/Llama-3.1-8B-Instruct-3-SHOT