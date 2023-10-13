#!/bin/bash

DATASET_PATH=$1
LOAD_MODEL_PATH=$2
LABELS_PATH=$3
OUTPUT_PATH=$4
ENV_SETUP=$5
OTHER_ARGS=$6

mkdir -p $CACHE
export SLURM_CPU_BIND="cores"
export HF_HOME=$CACHE
export HF_DATASETS_CACHE=$CACHE
export TRANSFORMERS_CACHE=$CACHE

source $ENV_SETUP

python pseudolabel.py \
    --dataset_path $DATASET_PATH \
    --config_path configs/config.yml \
    --model_path $LOAD_MODEL_PATH \
    --labels_path $LABELS_PATH \
    --output_path $OUTPUT_PATH \
    ${OTHER_ARGS}
