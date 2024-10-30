git clone https://github.com/Leonard907/ToolAlpaca.git
# cd ToolAlpaca
python process_toolalpaca.py
# clean up
mkdir -p sft_data
mv toolalpaca_eval_simulated.json sft_data/toolalpaca_eval_simulated_sharegpt.json
mv toolalpaca_eval_real.json sft_data/toolalpaca_eval_real_sharegpt.json
mv toolalpaca_train.json sft_data/toolalpaca_train_sharegpt.json
rm -rf ToolAlpaca