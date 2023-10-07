#!/bin/bash
#SBATCH -A dessn
#SBATCH -C gpu
#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH -c 32
#SBATCH -q shared
#SBATCH --gpus=1
#SBATCH --output=/pscratch/sd/h/helenqu/plasticc/self_training.log

export SLURM_CPU_BIND="cores"
export HF_HOME="/pscratch/sd/h/helenqu/huggingface_datasets_cache"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
python \
    /global/homes/h/helenqu/time_series_transformer/transformer_uda/self_training.py \
    --model_path $1 \
    $2
