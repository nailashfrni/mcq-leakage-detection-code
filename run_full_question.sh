#!/bin/bash

MODEL_NAME=$1
EVAL=$2

# Full
python run_evaluate.py \
    --input_file data/${EVAL}/${EVAL}.csv \
    --base_model_dir ${MODEL_NAME} \
    --eval ${EVAL} \
    --method semihalf \
    --prefix . \
    --mode full \
    --output_type csv \
    --compute_ppl True

python sample_cpt.py \
    --prefix . \
    --eval ${EVAL} \
    --df_file semihalf/result/${MODEL_NAME##*/}/${EVAL}/full_${EVAL}_${MODEL_NAME##*/}.csv \
    --model_dir ${MODEL_NAME}

python data_process.py \
    --data_dir data/${EVAL}/cpt_dataset_${EVAL}_${MODEL_NAME##*/}.json \
    --eval ${EVAL} \
    --save_dir permutation/data/${EVAL}

python data_process_quad.py \
    --data_dir data/${EVAL}/cpt_dataset_${EVAL}_${MODEL_NAME##*/}.json \
    --eval ${EVAL} \
    --save_dir permutation/data/${EVAL} \
    --fine_tune_type cpt