#!/bin/bash

# This is example of benchmark script for running lm_eval

MODEL_NAME=$1
BATCH_SIZE=4
TASKS=polish
NUM_FEWSHOT=0
DEVICE=cuda
VERBOSITY=DEBUG
WB_PROJECT=nondzu-lm-eval-v1 # leave empty to disable wandb

export NUMEXPR_MAX_THREADS=32
export CUDA_VISIBLE_DEVICES=0
export WANDB_ENTITY="nondzu"

# Check if user provided model name
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 model_name"
    exit 1
fi

# if few shot is 5 then set batch size to 1 to avoid OOM on GPU, remove if you have enough memory
if [ $NUM_FEWSHOT -eq 5 ]; then
    BATCH_SIZE=1
fi

echo "Batch size: ${BATCH_SIZE}"

# if last sign is a slash '/' remove it
if [ "${MODEL_NAME: -1}" == "/" ]; then
    MODEL_NAME=${MODEL_NAME::-1}
fi

# cut last part of model name to generate short model name 
SHORT_MODEL_NAME=$(echo ${MODEL_NAME} | rev | cut -d'/' -f 1 | rev)
OUTPUT_PATH="results/${SHORT_MODEL_NAME}-$(date +%Y%m%d-%H%M%S)-${NUM_FEWSHOT}shot"
echo "Model name: ${SHORT_MODEL_NAME}"

# Run benchmark and measure time of execution
if [ -n "$WB_PROJECT" ]; then
    time lm_eval --model hf --batch_size ${BATCH_SIZE} --model_args pretrained=${MODEL_NAME} --tasks ${TASKS} --num_fewshot ${NUM_FEWSHOT} --log_samples --device ${DEVICE} --verbosity ${VERBOSITY} --output_path ${OUTPUT_PATH} --wandb_args project=${WB_PROJECT},name=${SHORT_MODEL_NAME}
else
    time lm_eval --model hf --batch_size ${BATCH_SIZE} --model_args pretrained=${MODEL_NAME} --tasks ${TASKS} --num_fewshot ${NUM_FEWSHOT} --log_samples --device ${DEVICE} --verbosity ${VERBOSITY} --output_path ${OUTPUT_PATH}
fi

# to run on multiple GPUs use accelerate :
# time accelerate launch lm_eval --model hf --batch_size ${BATCH_SIZE} --model_args pretrained=${MODEL_NAME} --tasks ${TASKS} --num_fewshot ${NUM_FEWSHOT} --log_samples --device ${DEVICE} --verbosity ${VERBOSITY} --output_path ${OUTPUT_PATH}

# The second way of using accelerate for multi-GPU evaluation is when your model is too large to fit on a single GPU.
# passing parallelize=True to --model_args as follows:
# time lm_eval --model hf --batch_size ${BATCH_SIZE} --model_args pretrained=${MODEL_NAME},parallelize=True --tasks ${TASKS} --num_fewshot ${NUM_FEWSHOT} --log_samples --device ${DEVICE} --verbosity ${VERBOSITY} --output_path ${OUTPUT_PATH} --wandb_args project=lm-eval-harness-test,name=${SHORT_MODEL_NAME}   

echo "Benchmark finished. Results saved in: ${OUTPUT_PATH}"
