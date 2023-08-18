#!/bin/bash
#SBATCH -A dessn
#SBATCH -C gpu
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -c 128
#SBATCH -q regular
#SBATCH --gpus=4

LOAD_MODEL_PATH=$1
SAVE_MODEL=$2
WANDB_NAME=$3
LP_STEPS=$4
FT_STEPS=$5
MASK_BOOL=$6
WEIGHTS_BOOL=$7
OTHER_ARGS=$8

RANDOM_INIT=$([ "$1" == "None" ] && echo "--random_init" || echo "")
LOAD_MODEL=$([ "$1" != "None" ] && echo "--load_model $LOAD_MODEL_PATH" || "")
MASK=$([ "$MASK_BOOL" != "None" ] && echo "--mask" || echo "")
WEIGHTS=$([ "$WEIGHTS_BOOL" != "None" ] && echo "--class_weights" || echo "")
MIXED_PRECISION=$([ "$MASK_BOOL" != "None" ] && echo "--mixed_precision=bf16" || echo "")

echo $RANDOM_INIT
echo $LOAD_MODEL

export SLURM_CPU_BIND="cores"
export HF_HOME="/pscratch/sd/h/helenqu/huggingface_datasets_cache"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
accelerate launch --num_processes=4 \
    ${MIXED_PRECISION} \
     /global/homes/h/helenqu/time_series_transformer/transformer_uda/finetune_classification.py \
    --fourier_pe \
    --class_weights \
    --save_model $SAVE_MODEL \
    --num_lp_steps $LP_STEPS \
    --num_ft_steps $FT_STEPS \
    --wandb_name $WANDB_NAME \
    ${RANDOM_INIT} \
    ${LOAD_MODEL} \
    ${MASK} \
    ${WEIGHTS} \
    ${OTHER_ARGS}
    # --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_masked_no_distil_bf16_85k_steps/hf_model \
    # --load_model /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_75M_params_fourier/hf_model \
    # --context_length 170 \
    # --test_set_path /pscratch/sd/h/helenqu/plasticc/raw_test_with_labels \
    # --random_init
