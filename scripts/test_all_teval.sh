# export CUDA_VISIBLE_DEVICES=5
model=$1
model_path=$2
display_name=$3
is_api=$4

echo "Running evaluation with:
model: $model
path: $model_path
name: $display_name
api: $is_api"

echo "evaluating instruct on all ..."
python teval_eval.py --model "$model_path" --dataset_path data/instruct_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "instruct_${display_name}.json" --resume \
--eval instruct --prompt_type json --model_name "${display_name}" --tensor_parallel_size 1

echo "evaluating review ..."
python teval_eval.py --model "$model_path" --dataset_path data/review_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "review_str_${display_name}.json" --resume \
--eval review --prompt_type str --model_name "${display_name}" --tensor_parallel_size 1

echo "evaluating plan json ..."
python teval_eval.py --model "$model_path" --dataset_path data/plan_json_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "plan_json_${display_name}.json" --resume \
--eval plan --prompt_type json --model_name "${display_name}" --tensor_parallel_size 1

echo "evaluating plan str ..."
python teval_eval.py --model "$model_path" --dataset_path data/plan_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "plan_str_${display_name}.json" --resume \
--eval plan --prompt_type str --model_name "${display_name}" --tensor_parallel_size 1

echo "evaluating reason str ..."
python teval_eval.py --model "$model_path" --dataset_path data/reason_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "reason_str_${display_name}.json" --resume \
--eval reason --prompt_type str --model_name "${display_name}" --tensor_parallel_size 1

echo "evaluating retrieve str ..."
python teval_eval.py --model "$model_path" --dataset_path data/retrieve_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "retrieve_str_${display_name}.json" --resume \
--eval retrieve --prompt_type str --model_name "${display_name}" --tensor_parallel_size 1

echo "evaluating understand str ..."
python teval_eval.py --model "$model_path" --dataset_path data/understand_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "understand_str_${display_name}.json" --resume \
--eval understand --prompt_type str --model_name "${display_name}" --tensor_parallel_size 1

echo "evaluating RRU json ..."
python teval_eval.py --model "$model_path" --dataset_path data/reason_retrieve_understand_json_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "reason_retrieve_understand_json_${display_name}.json" --resume \
--eval rru --prompt_type json --model_name "${display_name}" --tensor_parallel_size 1

