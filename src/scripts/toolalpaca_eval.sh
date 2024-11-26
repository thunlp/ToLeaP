# Arguments
# $1: sharegpt data path
# $2: yaml config for llama factory
# $3: model name or path
echo "Input data path: $1"
echo "YAML config path: $2" 
echo "Model name/path: $3"

cp $1 llamafactory_data/toolalpaca.json

# Run evaluation with provided arguments
python toolalpaca_eval.py \
    --config $2 \
    --input_file $1 \
    --model_name $3