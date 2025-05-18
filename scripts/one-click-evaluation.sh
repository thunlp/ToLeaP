#!/bin/bash

# This file is to run this benchmark easily.
# Author: Zijun Song
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

MODEL_PATH=$1
IS_API=$2
GPU_NUM=$3
BATCH_SIZE=$4
INPUT_TOKENS=$5
OUTPUT_TOKENS=$6
if [ -n "$7" ]; then
  MODEL_NAME="$7"
else
  MODEL_NAME=$(basename "$MODEL_PATH")
fi

OUTPUT_FILE="${MODEL_NAME}_results.json"  

> $OUTPUT_FILE

echo "********** doing evaluation by $MODEL_NAME on TaskBench benchmark **********"
python taskbench_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API \
    --tensor_parallel_size $GPU_NUM \
    --batch_size $BATCH_SIZE \
    --max_model_len $INPUT_TOKENS \
    --model_name $MODEL_NAME \
    --max_output_tokens $OUTPUT_TOKENS | grep -oP '\{.*\}' >> $OUTPUT_FILE

echo "********** doing evaluation by $MODEL_NAME on SealTools benchmark **********"
python sealtools_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API \
    --tensor_parallel_size $GPU_NUM \
    --batch_size $BATCH_SIZE \
    --max_model_len $INPUT_TOKENS \
    --model_name $MODEL_NAME \
    --max_output_tokens $OUTPUT_TOKENS | grep -oP '\{.*\}' >> $OUTPUT_FILE

echo "********** doing evaluation by $MODEL_NAME on RoTBench benchmark **********"
python rotbench_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API \
    --tensor_parallel_size $GPU_NUM \
    --batch_size $BATCH_SIZE \
    --max_model_len $INPUT_TOKENS \
    --model_name $MODEL_NAME \
    --max_output_tokens $OUTPUT_TOKENS | grep -oP '\{.*\}' >> $OUTPUT_FILE

echo "********** doing evaluation by $MODEL_NAME on Glaive benchmark **********"
python glaive_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API \
    --tensor_parallel_size $GPU_NUM \
    --batch_size $BATCH_SIZE \
    --max_model_len $INPUT_TOKENS \
    --max_output_tokens $OUTPUT_TOKENS | grep -oP '\{.*\}' >> $OUTPUT_FILE

echo "********** doing evaluation by $MODEL_NAME on InjecAgent benchmark **********"
if [ "$IS_API" == "True" ]; then
    MODEL_TYPE=GPT
else
    MODEL_TYPE=OpenModel
fi
export OPENAI_API_KEY="your-open-api-kei"
python injecagent_eval.py \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_PATH \
    --use_cach | grep -oP '\{.*\}' >> $OUTPUT_FILE

echo "********** doing evaluation by $MODEL_NAME on T-Eval benchmark **********"
bash teval_eval.sh $MODEL_PATH $MODEL_NAME $IS_API $GPU_NUM

echo "********** doing evaluation by $MODEL_NAME on APIBank benchmark **********"
python apibank_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API \
    --tensor_parallel_size $GPU_NUM \
    --batch_size $BATCH_SIZE \
    --max_model_len $INPUT_TOKENS \
    --max_output_tokens $OUTPUT_TOKENS

echo "********** doing evaluation by $MODEL_NAME on BFCL benchmark **********"
bfcl generate \
    --model $MODEL_PATH \
    --test-category parallel,multiple,simple,parallel_multiple,java,javascript,irrelevance,multi_turn \
    --num-threads 1
bfcl evaluate \
    --model $MODEL_PATH
