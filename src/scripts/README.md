## 🛠️ Evaluation Method

**RoTBench Evaluation**  
RoTBench adapts three metrics, **Tool Selection (TS)**, **Parameter identification (PI)** and **Content filling (CF)**, to evalute funciton calling. Related methods are described in RoTBench_eval.py. To evaluate RoTBench, input should follow format:

**Tool Selection (TS)** represents whether agent can choose right function.
**Parameter identification (PI)** represents whether agent can fill right parameter name into function.
**Content filling (CF)** denotes whether agent can fill corrent content into corresponding parameters.

Input format include two files, **test_file** and **prediction_file**, which test_file should follow share_gpt file(.json) and prediction file should follow generated_predictions file format(.jsonl).

Run RoTBench Evaluation:
```
python src/scripts/RoTBench_eval.py --test_file PATH --answer_file PATH
```
 
 **Teval Evaluation**  (need fix, haven't finished yet)
 ```
 sh test_all_en.sh hf $HF_PATH $HF_MODEL_NAME $META_TEMPLATE
 eg: sh test_all_en.sh hf \workspace 3.1-8B-INS llama3 should work
 ```
