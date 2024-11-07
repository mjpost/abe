#!/bin/sh

#SBATCH --output=../slurm-logs/out-score-unit-test_eq_sent.log
#SBATCH --error=../slurm-logs/err-score-unit-test_eq_sent.log
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=20gb
#SBATCH --time=20:00:00

export HF_HOME=/fs/nexus-scratch/kravisan

#srun --time=6:00:00 python ensembling/simple-translate.py --input 'equivalent_sentences_check/eng_Latn.devtest' --output 'equivalent_sentences_check/fra_Latn_NLLB600.devtest'
srun --time=6:00:00 python ensembling/simple-translate.py --input 'equivalent_sentences_check/eng_Latn.devtest' --model 'facebook/m2m100_418M' --output 'equivalent_sentences_check/fra_Latn_M2M418.devtest'
