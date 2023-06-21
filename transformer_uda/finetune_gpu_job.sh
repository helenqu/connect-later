#!/bin/bash
#SBATCH -A dessn
#SBATCH -C gpu
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q regular
#SBATCH --gpus=1

export SLURM_CPU_BIND="cores"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
srun python /global/homes/h/helenqu/time_series_transformer/transformer_uda/finetune_classification.py \
    --context_length 170 \
    --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/finetuned_classification/pretrained_all_75M_params_randominit \
    --random_init
