#!/bin/bash

MODEL_NAME=$1
EVAL_NAME=$2
USER_HF=$3

INPUT_FILE="data/$EVAL_NAME/cpt_dataset_${EVAL_NAME}_${MODEL_NAME##*/}.json"

# # Semihalf
python run_evaluate.py \
    --input_file $INPUT_FILE \
    --base_model_dir $MODEL_NAME \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --eval $EVAL_NAME \
    --method semihalf \
    --prefix . \
    --fine_tune_type cpt \
    --mode semihalf \
    --output_type json \
    --compute_ppl False \
    --epoch 1

python run_evaluate.py \
    --input_file $INPUT_FILE \
    --base_model_dir $MODEL_NAME \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --eval $EVAL_NAME \
    --method semihalf \
    --prefix . \
    --fine_tune_type cpt \
    --mode semihalf \
    --output_type json \
    --compute_ppl False \
    --epoch 5

python run_evaluate.py \
    --input_file $INPUT_FILE \
    --base_model_dir $MODEL_NAME \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --eval $EVAL_NAME \
    --method semihalf \
    --prefix . \
    --fine_tune_type cpt \
    --mode semihalf \
    --output_type json \
    --compute_ppl False \
    --epoch 10

# # Ngram
python run_evaluate.py \
    --input_file $INPUT_FILE \
    --base_model_dir $MODEL_NAME \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --eval $EVAL_NAME \
    --method ngram \
    --prefix . \
    --fine_tune_type cpt \
    --epoch 1

python run_evaluate.py \
    --input_file $INPUT_FILE \
    --base_model_dir $MODEL_NAME \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --eval $EVAL_NAME \
    --method ngram \
    --prefix . \
    --fine_tune_type cpt \
    --epoch 5

python run_evaluate.py \
    --input_file $INPUT_FILE \
    --base_model_dir $MODEL_NAME \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --eval $EVAL_NAME \
    --method ngram \
    --prefix . \
    --fine_tune_type cpt \
    --epoch 10

# Permutation
## Inference Logprob
python inference_logprob.py \
    --base_model_dir $MODEL_NAME \
    --prefix . \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --fine_tune_type cpt \
    --eval $EVAL_NAME \
    --checkpoint_epoch 1

python inference_logprob.py \
    --base_model_dir $MODEL_NAME \
    --prefix . \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --fine_tune_type cpt \
    --eval $EVAL_NAME \
    --checkpoint_epoch 5

python inference_logprob.py \
    --base_model_dir $MODEL_NAME \
    --prefix . \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --fine_tune_type cpt \
    --eval $EVAL_NAME \
    --checkpoint_epoch 10

## Outliers
python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 24 \
    --prefix . \
    --model_name $MODEL_NAME \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-1.json

python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 24 \
    --prefix . \
    --model_name $MODEL_NAME \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-5.json

python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 24 \
    --prefix . \
    --model_name $MODEL_NAME \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-10.json

# Permutation-R
python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 24 \
    --prefix . \
    --model_name $MODEL_NAME \
    --is_selected True \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-1.json

python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 24 \
    --prefix . \
    --model_name $MODEL_NAME \
    --is_selected True \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-5.json

python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 24 \
    --prefix . \
    --model_name $MODEL_NAME \
    --is_selected True \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-10.json

## Permutation-Q
## Inference Logprob
python inference_logprob.py \
    --base_model_dir $MODEL_NAME \
    --prefix . \
    --permutations_data_dir permutation/data/$EVAL_NAME/quad_permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --fine_tune_type cpt \
    --eval $EVAL_NAME \
    --checkpoint_epoch 1

python inference_logprob.py \
    --base_model_dir $MODEL_NAME \
    --prefix . \
    --permutations_data_dir permutation/data/$EVAL_NAME/quad_permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --fine_tune_type cpt \
    --eval $EVAL_NAME \
    --checkpoint_epoch 5

python inference_logprob.py \
    --base_model_dir $MODEL_NAME \
    --prefix . \
    --permutations_data_dir permutation/data/$EVAL_NAME/quad_permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --adapter_dir ${USER_HF}/cpt_${MODEL_NAME##*/}_$EVAL_NAME \
    --fine_tune_type cpt \
    --eval $EVAL_NAME \
    --checkpoint_epoch 10

## Outliers
python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/quad_permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 12 \
    --prefix . \
    --model_name $MODEL_NAME \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/quad_logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-1.json

python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/quad_permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 12 \
    --prefix . \
    --model_name $MODEL_NAME \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/quad_logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-5.json

python get_outlier.py \
    --permutations_data_dir permutation/data/$EVAL_NAME/quad_permutations_data_cpt_dataset_${MODEL_NAME##*/}.json \
    --method not_shuffled \
    --eval $EVAL_NAME \
    --permutation_num 12 \
    --prefix . \
    --model_name $MODEL_NAME \
    --logprobs_dir permutation/result/${MODEL_NAME##*/}/$EVAL_NAME/logprobs/quad_logprobs_${MODEL_NAME##*/}_${EVAL_NAME}_cpt-cp-epoch-10.json