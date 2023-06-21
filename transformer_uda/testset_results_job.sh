#!/bin/bash
#SBATCH -A dessn
#SBATCH -C gpu
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -q regular

export SLURM_CPU_BIND="cores"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
python /global/homes/h/helenqu/time_series_transformer/transformer_uda/testset_results.py \
    --pretrained_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/finetuned_classification/pretrained_all_75M_params_randominit/hf_model \
    --model_config /global/homes/h/helenqu/time_series_transformer/transformer_uda/configs/bigger_model_hyperparameters.yml
    # --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_75M_params/hf_model \
    # --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/hf_model_500k_pretrained \
    # --num_epochs 200 \
