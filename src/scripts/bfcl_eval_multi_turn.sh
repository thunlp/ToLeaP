# Use bfcl for evaluation of multi-turn conversations

num_gpus=${2:-1}
gpu_memory_utilization=${3:-0.8}

echo "Running bfcl generate..."
bfcl generate --model $1 --test-category multi_turn --temperature 0 --num-gpus $num_gpus --gpu-memory-utilization $gpu_memory_utilization --result-dir $(pwd) > /dev/null 2>&1
echo "Generation done."
echo "Running bfcl evaluate..."
bfcl evaluate --model $1 --test-category multi_turn
python bfcl_multi_turn_converter.py --model $1
