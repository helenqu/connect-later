#!/bin/bash
#SBATCH -A dessn
#SBATCH -C gpu
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH -c 128
#SBATCH -q regular
#SBATCH --gpus=4

export SLURM_CPU_BIND="cores"
export HF_HOME="/pscratch/sd/h/helenqu/huggingface_datasets_cache"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    /global/homes/h/helenqu/time_series_transformer/transformer_uda/self_training.py \
    --model_path $1
