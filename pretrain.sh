#!/bin/bash

DATA_PATH=$1
SAVE_MODEL=$2
WANDB_NAME=$3
ENV_SETUP=$4
NUM_GPUS=$5
OTHER_ARGS=$6

mkdir -p $CACHE
export SLURM_CPU_BIND="cores"
export HF_HOME=$CACHE
export HF_DATASETS_CACHE=$CACHE
export TRANSFORMERS_CACHE=$CACHE

source $ENV_SETUP

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    connect_later/pretrain.py \
    --config_path configs/config.yml \
    --dataset_path $DATA_PATH \
    --wandb_name $WANDB_NAME \
    --save_model $SAVE_MODEL \
    --num_steps 60_000 \
    ${OTHER_ARGS} \

