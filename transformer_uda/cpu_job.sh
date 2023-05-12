#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q shared

export SLURM_CPU_BIND="cores"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
python /global/homes/h/helenqu/time_series_transformer/transformer_uda/transformer_uda/huggingface_informer.py \
    --data_dir /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples \
    --load_checkpoint /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/checkpoints/checkpoint_2023-05-08_11:30:50_epoch_100 \
    --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_500k_context_100_pred_50 \
    --num_epochs 50
