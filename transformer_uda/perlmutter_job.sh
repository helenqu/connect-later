#!/bin/bash
#SBATCH -A m1727
#SBATCH -C gpu
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q shared
#SBATCH --gpus-per-node=1

export SLURM_CPU_BIND="cores"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
srun python /global/homes/h/helenqu/time_series_transformer/transformer_uda/transformer_uda/huggingface_informer.py --data_dir /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_500k_context_100_pred_50 --num_epochs 150
