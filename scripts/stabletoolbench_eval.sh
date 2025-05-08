export TOOLBENCH_KEY="your-toolbench-key"
export PYTHONPATH=./
export OPEN_MODEL="baichuan-inc/Baichuan2-7B-Chat"
export MODEL_NAME="Baichuan2-7B-Chat"

TEST_GROUPS=("G1_instruction" "G1_category" "G1_tool" "G2_instruction" "G2_category" "G3_instruction")

for GROUP in "${TEST_GROUPS[@]}"; do
    echo "********** doing evaluation by $MODEL_NAME on $GROUP dataset **********"

    python ../src/benchmark/stabletoolbench/toolbench/inference/qa_pipeline_multithread.py \
        --tool_root_dir ../data/stabletoolbench/tools \
        --backbone_model toolllama_vllm \
        --model_path $OPEN_MODEL \
        --input_query_file "../data/stabletoolbench/solvable_queries/test_instruction/$GROUP.json" \
        --toolbench_key $TOOLBENCH_KEY \
        --output_path "../results/stabletoolbench/$MODEL_NAME/$GROUP" \
        --tensor_parallel_size 1

    python stabletoolbench_eval.py \
        --model_name $MODEL_NAME \
        --group $GROUP \
        --eval_model "deepseek-r1"
done

