# Arguments
# --eval_file: the path to the evaluation file
# --model: the model to use for evaluation

# Assuming evaluation files are ready
python toolalpaca_eval.py --eval_file ../data/sft_data/toolalpaca_eval_real_sharegpt.json --model meta-llama/Llama-3.1-8B-Instruct
python toolalpaca_eval.py --eval_file ../data/sft_data/toolalpaca_eval_simulated_sharegpt.json --model meta-llama/Llama-3.1-70B-Instruct