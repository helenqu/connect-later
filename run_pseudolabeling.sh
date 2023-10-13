#!/bin/bash

source env_vars.sh

DATASET_PATH=$1
LOAD_MODEL_PATH=$2
LABELS_PATH=$3
OUTPUT_PATH=$4
OTHER_ARGS=$5

NUM_GPUS=1

mkdir -p $LOGDIR

jid=$(sbatch \
--parsable \
${cluster_info} \
-c 32 \
${queue_info} \
--gpus=${NUM_GPUS} \
-n 1 \
-t 12:00:00 \
--output ${LOGDIR}/${WANDB_NAME}.log \
pseudolabel_job.sh  ${DATASET_PATH} ${LOAD_MODEL_PATH} ${LABELS_PATH} ${OUTPUT_PATH} ${ENV_SETUP} "${OTHER_ARGS}")
echo pseudolabel_job.sh  ${DATASET_PATH} ${LOAD_MODEL_PATH} ${LABELS_PATH} ${OUTPUT_PATH} ${ENV_SETUP} "${OTHER_ARGS}"

echo "Submitted job ${jid}"
