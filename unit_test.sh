#!/bin/sh

#SBATCH --output=../slurm-logs/out-score-unit-test.log
#SBATCH --error=../slurm-logs/err-score-unit-test.log
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=20gb
#SBATCH --time=20:00:00

export HF_HOME=/fs/nexus-scratch/kravisan

#srun --time=06:00:00 python ensembling/ensemble.py -i 'input-test.jsonl' -o 'output_flores_devtest_engtofr' -w 0.5 0.5 --score
srun --time=6:00:00 python ensembling/solo_scores.py --input 'output_flores_devtest_engtofr.jsonl'
