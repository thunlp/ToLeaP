#!/usr/bin/env bash

# This file is to run SkyThought benchmark easily.
# Author: Zijun Song
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

# —— 在这里填入你要测试的模型路径 —— 
models=(
  "/share_data/data2/workhome/chenhaotian/models/models--Team-ACE--ToolACE-2-8B/snapshots/91c918927e687541cc3f8dd4dbed42014efbc147"
  "/share_data/data2/workhome/chenhaotian/models/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
  "/share_data/data2/workhome/chenhaotian/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
  "/share_data/data2/workhome/chenhaotian/models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
  "/share_data/data2/workhome/chenhaotian/models/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
  "/share_data/data2/workhome/chenhaotian/models/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
  # 如果有新模型，直接在这里加一行
)

# —— 在这里填入你要测试的任务名称 —— 
tasks=(
  "math500"
#   "numina"
  "numina_amc_aime"
  "numina_math"
  "numina_olympiads"
  "taco"
  "olympiadbench_math_en"
  "aime24_sky"
  "aime24"
  "aime25"
  "amc23"
  "livecodebench_medium"
  "livecodebench"
  "livecodebench_easy"
  "livecodebench_hard"
  "arc_c"
  "apps"
  "mmlu"
  "mmlu_pro"
  "gsm8k"
  "minervamath"
  "gpqa_diamond"
)

backend="vllm"
backend_args="tensor_parallel_size=4"

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    echo "=== Evaluating $(basename "$model") on $task ==="
    skythought evaluate \
      --model "$model" \
      --backend "$backend" \
      --backend-args "$backend_args" \
      --task "$task" \
      --overwrite \
      || echo "[ERROR] Failed: $(basename "$model") | $task"
  done
done

echo "✅ All done!"
