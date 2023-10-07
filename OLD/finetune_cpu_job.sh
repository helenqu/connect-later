#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -q shared

export SLURM_CPU_BIND="cores"
export HF_HOME="/pscratch/sd/h/helenqu/huggingface_datasets_cache"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
accelerate launch --num_processes=1 \
    --mixed_precision=bf16 \
     /global/homes/h/helenqu/time_series_transformer/transformer_uda/finetune_classification.py \
    --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_masked_no_distil_bf16_85k_steps/hf_model \
    --fourier_pe \
    --mask \
    --class_weights \
    --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/finetuned_classification/pretrained_masked_weighted_lpft_augment \
    --dataset_path /pscratch/sd/h/helenqu/plasticc/raw_baseline_self_labeled_test \
    --wandb_name baseline_self_labeled \
    --random_init \
    --num_lp_steps 0 \
    --num_ft_steps 100
