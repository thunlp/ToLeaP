# Arguments
# $1: the path to the evaluation file
# $2: (optional) the path to the data file

# Run evaluation with provided arguments
if [ -z "$2" ]; then
    python taskbench_eval.py --result_path "$1"
else
    python taskbench_eval.py --result_path "$1" --src_data_path "$2"
fi