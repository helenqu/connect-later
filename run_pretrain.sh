#!/bin/bash

source env_vars.sh

DATA_PATH=$1
SAVE_MODEL_PATH=$2
WANDB_NAME=$3
OTHER_ARGS=$4

NUM_GPUS=4

mkdir -p $LOGDIR

jid=$(sbatch \
    --parsable \
    ${cluster_info} \
    -c 128 \
    ${queue_info} \
    --gpus=${NUM_GPUS} \
    -n 1 \
    -t 12:00:00 \
    --output ${LOGDIR}/${WANDB_NAME}.log \
    pretrain.sh ${DATA_PATH} ${SAVE_MODEL_PATH} ${WANDB_NAME} ${ENV_SETUP} ${NUM_GPUS} "${OTHER_ARGS}")
echo "pretrain.sh ${DATA_PATH} ${SAVE_MODEL_PATH} ${WANDB_NAME} ${ENV_SETUP} ${NUM_GPUS} ${OTHER_ARGS}"

echo "Submitted job ${jid}"
