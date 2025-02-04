#!/bin/bash

MODEL_PATH=$1

# TaskBench
python taskbench_eval.py \
    --model $MODEL_PATH \
    --data_path /home/test/test03/szj/BodhiAgent-main/src/data/sft_data/TaskBench/taskbench_data_huggingface.json \
    --is_api False
python taskbench_eval.py \
    --model $MODEL_PATH \
    --data_path /home/test/test03/szj/BodhiAgent-main/src/data/sft_data/TaskBench/taskbench_data_dailylifeapis.json \
    --is_api False
python taskbench_eval.py \
    --model $MODEL_PATH \
    --data_path /home/test/test03/szj/BodhiAgent-main/src/data/sft_data/TaskBench/taskbench_data_multimedia.json \
    --is_api False

# RoTBench
python sealtools_eval.py \
    --model $MODEL_PATH \
    --is_api False

# SealTools
python rotbench_eval.py \
    --model $MODEL_PATH \
    --is_api False

# BFCL
bfcl generate \
    --model $MODEL_PATH \
    --test-category parallel,multiple,simple,parallel_multiple,java,javascript,irrelevance,multi_turn \
    --num-threads 1
bfcl evaluate \
    --model $MODEL_PATH

# T-Eval
bash test_all_teval.sh vllm $MODEL_PATH QwQ-32B-Preview False

# InjecAgent
python3 injecagent_eval.py \
    --model_type OpenModel \
    --model_name $MODEL_PATH \
    --use_cach
