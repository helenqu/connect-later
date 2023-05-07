#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q shared

module load python
python /global/homes/h/helenqu/transformer_uda/transformer_uda/data_to_examples.py $1
