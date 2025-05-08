#!/usr/bin/env bash

# This file is to run SkyThought benchmark easily.
# Author: Zijun Song
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

models=(
  "meta/Llama-3.2-1B"
  "meta/Llama-3.1-8B-Instruct"
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
backend_args="tensor_parallel_size=1"

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
