set -e

mkdir -p result

read -p "请输入 model-name: " model_name
model_folder="result/${model_name//\//_}"
mkdir -p "$model_folder"

mv BFCL*.json "$model_folder" 2>/dev/null || echo "No BFCL*.json files found."

cd bfcl/eval_checker || { echo "Directory bfcl/eval_checker not found."; exit 1; }

python eval_runner.py
