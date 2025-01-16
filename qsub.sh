#!/usr/bin/env bash

#$ -l h_rt=5:59:0,gpu=1
#$ -q gpu_short.q,gpu.q@@rtx
#$ -cwd
#$ -o logs/
#$ -j y
#$ -cwd

source ~/.bashrc

cd /exp/rwicks/ensemble24

MODEL_ONE=$1
MODEL_TWO=$2
LANGUAGE_PAIR=$3
TESTSET=$4

conda deactivate
conda activate ensemble311

bash translation.sh $MODEL_ONE $MODEL_TWO $LANGUAGE_PAIR $TESTSET