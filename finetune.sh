#!/bin/bash

LOAD_MODEL_PATH=$1
SAVE_MODEL_PATH=$2
DATA_PATH=$3
WANDB_NAME=$4
ENV_SETUP=$5
NUM_GPUS=$6
OTHER_ARGS=$7

RANDOM_INIT=$([ "$1" == "None" ] && echo "--random_init" || echo "")
LOAD_MODEL=$([ "$1" != "None" ] && echo "--load_model $LOAD_MODEL_PATH" || "")

mkdir -p $CACHE
export SLURM_CPU_BIND="cores"
export HF_HOME=$CACHE
export HF_DATASETS_CACHE=$CACHE
export TRANSFORMERS_CACHE=$CACHE

source $ENV_SETUP

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    finetune_classification.py \
    --config_path configs/config.yml \
    --dataset_path $DATA_PATH \
    --save_model $SAVE_MODEL_PATH \
    --num_lp_steps 20_000 \
    --num_ft_steps 10_000 \
    --wandb_name $WANDB_NAME \
    ${RANDOM_INIT} \
    ${LOAD_MODEL} \
    ${OTHER_ARGS}
#TODO: Add weights and test set path to other args
