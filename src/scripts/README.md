## 🛠️ Evaluation Method

**RoTBench Evaluation**  
RoTBench adapts three metrics, **Tool Selection (TS)**, **Parameter identification (PI)** and **Content filling (CF)**, to evalute funciton calling. Related methods are described in RoTBench_eval.py. To evaluate RoTBench, input should follow format:
```
Input format and Expected Output format
.....
```
 
 **Teval Evaluation**  (need fix, haven't finished yet)
 ```
 sh test_all_en.sh hf $HF_PATH $HF_MODEL_NAME $META_TEMPLATE
 eg: sh test_all_en.sh hf \workspace 3.1-8B-INS llama3 should work
 ```
