git clone https://github.com/Leonard907/ToolAlpaca.git
# cd ToolAlpaca
python process_toolalpaca.py
# clean up
mkdir toolalpaca_data
mv toolalpaca_eval_simulated.json toolalpaca_data/toolalpaca_eval_simulated_sharegpt.json
mv toolalpaca_eval_real.json toolalpaca_data/toolalpaca_eval_real_sharegpt.json
mv toolalpaca_train.json toolalpaca_data/toolalpaca_train_sharegpt.json
rm -rf ToolAlpaca