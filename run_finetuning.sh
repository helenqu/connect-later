#!/bin/bash

source env_vars.sh

LOAD_MODEL_PATH=$1
SAVE_MODEL_PATH=$2
DATA_PATH=$3
WANDB_NAME=$4
OTHER_ARGS=$5

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
finetune.sh ${LOAD_MODEL_PATH} ${SAVE_MODEL_PATH} ${DATA_PATH} ${WANDB_NAME}  ${ENV_SETUP} ${NUM_GPUS} "${OTHER_ARGS}")
echo "finetune.sh ${LOAD_MODEL_PATH} ${SAVE_MODEL_PATH} ${DATA_PATH} ${WANDB_NAME}  ${ENV_SETUP} ${NUM_GPUS} ${OTHER_ARGS}"

echo "Submitted job ${jid}"
