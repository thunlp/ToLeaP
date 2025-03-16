# export CUDA_VISIBLE_DEVICES=2,3
model_path=$1
display_name=$2
is_api=$3
tensor_parallel_size=${4:-1}  # 
gpu_memory_utilization=${5:-0.9}  # 
use_special_tokens=${6:-false}  

echo "Running evaluation with:
path: $model_path
name: $display_name
api: $is_api
tensor_parallel_size: $tensor_parallel_size
gpu_memory: $gpu_memory_utilization
special_tokens: $use_special_tokens"

special_tokens_args=""
if [ "$use_special_tokens" = "true" ]; then
    special_tokens_args="--special_tokens"
fi

echo "evaluating instruct on all ..."
python teval_eval.py --model "$model_path" --dataset_path data/instruct_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "instruct_${display_name}.json" --resume \
--eval instruct --prompt_type json --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

echo "evaluating review ..."
python teval_eval.py --model "$model_path" --dataset_path data/review_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "review_str_${display_name}.json" --resume \
--eval review --prompt_type str --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

echo "evaluating plan json ..."
python teval_eval.py --model "$model_path" --dataset_path data/plan_json_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "plan_json_${display_name}.json" --resume \
--eval plan --prompt_type json --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

echo "evaluating plan str ..."
python teval_eval.py --model "$model_path" --dataset_path data/plan_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "plan_str_${display_name}.json" --resume \
--eval plan --prompt_type str --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

echo "evaluating reason str ..."
python teval_eval.py --model "$model_path" --dataset_path data/reason_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "reason_str_${display_name}.json" --resume \
--eval reason --prompt_type str --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

echo "evaluating retrieve str ..."
python teval_eval.py --model "$model_path" --dataset_path data/retrieve_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "retrieve_str_${display_name}.json" --resume \
--eval retrieve --prompt_type str --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

echo "evaluating understand str ..."
python teval_eval.py --model "$model_path" --dataset_path data/understand_str_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "understand_str_${display_name}.json" --resume \
--eval understand --prompt_type str --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

echo "evaluating RRU json ..."
python teval_eval.py --model "$model_path" --dataset_path data/reason_retrieve_understand_json_v2.json --is_api $is_api \
--out_dir "work_dirs/${display_name}/" --out_name "reason_retrieve_understand_json_${display_name}.json" --resume \
--eval rru --prompt_type json --model_name "${display_name}" \
--tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization $special_tokens_args

