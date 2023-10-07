#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -q shared

export SLURM_CPU_BIND="cores"
export HF_HOME="/pscratch/sd/h/helenqu/huggingface_datasets_cache"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
python /global/homes/h/helenqu/time_series_transformer/transformer_uda/transformer_uda/huggingface_informer.py \
    --data_dir /pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples \
    --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_masked \
    --fourier_pe \
    --mask \
    --num_steps 10_000
    # --load_checkpoint /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/checkpoints/checkpoint_2023-05-08_11:30:50_epoch_100 \
    # --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_500k_context_100_pred_50 \
