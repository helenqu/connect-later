#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q shared

module load python
python /global/homes/h/helenqu/time_series_transformer/transformer_uda/transformer_uda/raw_data_to_examples.py $1
