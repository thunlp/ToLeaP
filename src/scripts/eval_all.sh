export CUDA_VISIBLE_DEVICES=4,5,6,7
model=$1
display_name=$2

echo "开始评估，使用以下参数:
模型: $model
名称: $display_name"

echo "正在评估多个任务..."
python evaluate.py --model "$model" \
--dataset_path "[data/instruct_v2.json,\
data/review_str_v2.json,\
data/plan_json_v2.json,\
data/plan_str_v2.json,\
data/reason_str_v2.json,\
data/retrieve_str_v2.json,\
data/understand_str_v2.json,\
data/reason_retrieve_understand_json_v2.json]" \
--out_dir "work_dirs/${display_name}/" \
--out_name "[instruct_${display_name}.json,\
review_str_${display_name}.json,\
plan_json_${display_name}.json,\
plan_str_${display_name}.json,\
reason_str_${display_name}.json,\
retrieve_str_${display_name}.json,\
understand_str_${display_name}.json,\
reason_retrieve_understand_json_${display_name}.json]" \
--eval "[instruct,review,plan,plan,reason,retrieve,understand,rru]" \
--prompt_type "[json,str,json,str,str,str,str,json]" \