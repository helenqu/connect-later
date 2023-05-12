#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q shared

export SLURM_CPU_BIND="cores"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
python /global/homes/h/helenqu/time_series_transformer/transformer_uda/finetune_classification.py \
    --context_length 170 \
    --save_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/finetuned_classification/pretrained_500k_context_170_class_weights \
    --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_500k_context_170/hf_model \
    # --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/hf_model_500k_pretrained \
    # --num_epochs 200 \
