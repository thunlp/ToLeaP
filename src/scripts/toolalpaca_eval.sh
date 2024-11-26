# Arguments
# $1: the input data path
# $2: (optional) the path to the data file

cp $1 llamafactory_data/toolalpaca.json

# Run evaluation with provided arguments
if [ -z "$2" ]; then
    python toolalpaca_eval.py --eval_file "$1"
else
    python toolalpaca_eval.py --eval_file "$1" --data_file "$2"
fi