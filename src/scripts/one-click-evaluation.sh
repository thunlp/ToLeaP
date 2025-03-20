#!/bin/bash

MODEL_PATH=$1
IS_API=$2
MODEL_NAME=$(basename $MODEL_PATH)  
OUTPUT_FILE="${MODEL_NAME}_results.json"  

> $OUTPUT_FILE

# TaskBench
python taskbench_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API | grep -oP '\{.*\}' >> $OUTPUT_FILE

# SealTools
python sealtools_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API | grep -oP '\{.*\}' >> $OUTPUT_FILE

# RoTBench
python rotbench_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API | grep -oP '\{.*\}' >> $OUTPUT_FILE

# Glaive
python glaive_eval.py \
    --model $MODEL_PATH \
    --is_api $IS_API | grep -oP '\{.*\}' >> $OUTPUT_FILE

# InjecAgent
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

# BFCL
bfcl generate \
    --model $MODEL_PATH \
    --test-category parallel,multiple,simple,parallel_multiple,java,javascript,irrelevance,multi_turn \
    --num-threads 1
bfcl evaluate \
    --model $MODEL_PATH >> "BFCL_results.txt"
python standard_bfcl.py | grep -oP '\{.*\}' >> $OUTPUT_FILE

# Skythought
bash one-click-sky.sh $MODEL_PATH | grep -oP '\{.*\}' >> $OUTPUT_FILE

# T-Eval
cd T-Eval
bash test_all_teval.sh $MODEL_PATH $MODEL_NAME $IS_API
bash eval_all.sh $MODEL_NAME $MODEL_NAME
cd ..
python standard_teval.py T-Eval/work_dirs/$MODEL_NAME/${MODEL_NAME}_-1_.json
