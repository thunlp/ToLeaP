#!/bin/bash

MODEL_PATH=""

# parse
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# if model path provided
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 --model <model_path>"
    exit 1
fi

# RoTBench
python rotbench_eval.py --model "$MODEL_PATH" --raw_data_path /bjzhyai03/workhome/songzijun/BodhiAgent-main/src/data/eval_data/RoTBench/First_turn/clean.json
python rotbench_eval.py --model "$MODEL_PATH" --raw_data_path /bjzhyai03/workhome/songzijun/BodhiAgent-main/src/data/eval_data/RoTBench/First_turn/union.json
python rotbench_eval.py --model "$MODEL_PATH" --raw_data_path /bjzhyai03/workhome/songzijun/BodhiAgent-main/src/data/eval_data/RoTBench/First_turn/medium.json
python rotbench_eval.py --model "$MODEL_PATH" --raw_data_path /bjzhyai03/workhome/songzijun/BodhiAgent-main/src/data/eval_data/RoTBench/First_turn/heavy.json
python rotbench_eval.py --model "$MODEL_PATH" --raw_data_path /bjzhyai03/workhome/songzijun/BodhiAgent-main/src/data/eval_data/RoTBench/First_turn/slight.json

# SealTools
python sealtools_eval.py --model "$MODEL_PATH" 

# TaskBench
python taskbench_eval.py --model "$MODEL_PATH" --data_path /bjzhyai03/workhome/songzijun/oldversion/src/data/sft_data/TaskBench/taskbench_data_dailylifeapis.json
python taskbench_eval.py --model "$MODEL_PATH" --data_path /bjzhyai03/workhome/songzijun/oldversion/src/data/sft_data/TaskBench/taskbench_data_multimedia.json
python taskbench_eval.py --model "$MODEL_PATH" --data_path /bjzhyai03/workhome/songzijun/oldversion/src/data/sft_data/TaskBench/taskbench_data_huggingface.json

# T-Eval
sh test_all_en.sh hf "$MODEL_PATH" $HF_MODEL_NAME $META_TEMPLATE # TODO
