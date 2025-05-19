#!/bin/bash

MODEL_NAME=$1
EVAL=$2
USER_HF=$3

python run_cpt.py \
    --model_name ${MODEL_NAME} \
    --eval ${EVAL} \
    --data_files cpt/data/${EVAL}/cpt_dataset_${EVAL}_${MODEL_NAME##*/}_train.json \
    --num_epochs 11 \
    --hf_repo ${USER_HF}/cpt_${MODEL_NAME##*/}_${EVAL}