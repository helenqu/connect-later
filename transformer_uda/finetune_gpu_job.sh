#!/bin/bash
#SBATCH -A dessn
#SBATCH -C gpu
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q shared
#SBATCH --gpus=1

export SLURM_CPU_BIND="cores"
export HF_HOME="/pscratch/sd/h/helenqu/huggingface_datasets_cache"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
accelerate launch --num_processes=1 \
    --mixed_precision=bf16 \
     /global/homes/h/helenqu/time_series_transformer/transformer_uda/finetune_classification.py \
    --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_masked_no_distil_bf16/hf_model \
    --fourier_pe \
    --mask \
    # --balance
    # --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_75M_params_fourier/hf_model \
    # --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/finetuned_classification/pretrained_all_75M_params_fourier_pretrained_weights \
    # --context_length 170 \
    # --test_set_path /pscratch/sd/h/helenqu/plasticc/raw_test_with_labels \
