git clone https://github.com/Leonard907/ToolAlpaca.git
cd ToolAlpaca
python build_dataset.py -api data/train_data.json -out ../toolalpaca_train.json
python build_dataset.py -api data/eval_simulated.json -out ../toolalpaca_val_sim.json -eval
python build_dataset.py -api data/eval_real.json -out ../toolalpaca_val_real.json -eval
cd ..
python process_toolalpaca.py
# clean up
rm -rf ToolAlpaca
mkdir toolalpaca_data
mv toolalpaca_train.json toolalpaca_data/
mv toolalpaca_val_sim.json toolalpaca_data/
mv toolalpaca_val_real.json toolalpaca_data/