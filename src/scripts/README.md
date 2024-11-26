## 🛠️ Evaluation Method

<font size="5">**RoTBench Evaluation**</font>  

$\qquad$ RoTBench adapts three metrics, **Tool Selection (TS)**, **Parameter identification (PI)** and **Content filling (CF)**, to evalute funciton calling. Related methods are described in RoTBench_eval.py. To evaluate RoTBench, input should follow format:

$\qquad$ **Tool Selection (TS)** represents whether agent can choose right function.
**Parameter identification (PI)** represents whether agent can fill right parameter name into function.
**Content filling (CF)** denotes whether agent can fill corrent content into corresponding parameters.

$\qquad$ Input format include two files, **test_file** and **prediction_file**, which test_file should follow share_gpt file(.json) and prediction file should follow generated_predictions file format(.jsonl).

$\qquad$ Run RoTBench Evaluation:
```
python src/scripts/RoTBench_eval.py --test_file PATH --answer_file PATH
```

<font size="5"> **APIBANK Evaluation**</font>  
 
$\qquad$ APIBANK adapts two metrics, **Accuarcy of Tool Selection** and **Rouge** to evaluate function calling.  
$\qquad$ Run APIBANK Evaluation:
```
python src/scripts/APIBANK_eval.py.py --test_file PATH --answer_file PATH
```
<font size="5">**Toollens**</font>  
$\qquad$ Toollen apdats three metrics to evaluate tool retrieval abilities, as same with tool calling abilities in this scenario:**Recall@K**, **NDCG@K** and **COMP@K**.

$\qquad$ **Recall@K** metric measures how many relevant items were successfully retrieved from the entire dataset.  
$\qquad$ **NDCG@K** Normalized Discounted Cumulative Gain@K (NDCG@K) metric measures the system's ability to sort items based on relevance.  
$\qquad$ **COMP@K** measure whether the top-𝐾 retrieved tools form a complete set with respect to the ground-truth set.  
$\qquad$ Run Toollens Evaluation:
```
#这个Toollens比较复杂 为了预防看不懂，我加了中文注释，topk是一个element为int type的list, 来截断检索到的k个工具；Ground_truth 是一个list 代表label, 即有哪些工具可以解决这个问题；pred_i 是另外一个list 代表模型预测出来用哪些工具来解决query。这玩意原理搞明白了，现在还用不了，主要存在同时多个工具调用，无测试样例等问题。
python src/scripts/Toollens_eval.py --topk List[Int] --ground_truth_u_i List --pred_i List
```


 **BFCL Evaluation**   
 ```

 ```

 **Teval Evaluation**  
 ```
python teval_eval.py
 ```

**ToolAlpaca Evaluation**  
Follow instructions in LLaMA-Factory to first install the repo, then run the evaluation.
```
bash toolalpaca_eval.sh $INPUT $CONFIG $MODEL
# example below
bash toolalpaca_eval.sh sharegpt_data.json toolalpaca.yml meta-llama/Meta-Llama-3.1-8B-Instruct
```
`$INPUT` Path of the sharegpt toolalpaca file.
`$CONFIG` Path of the yaml config file for LLaMA-Factory.
`$MODEL` Model name or path.

ToolAlpaca measures process and response correctness using GPT-4. Template for evaluation can be found in `utils/template.py`. The final outcome is the percentage of "yes" outputs for both process and response correctness.

**TaskBench Evaluation**  
Follow instructions in LLaMA-Factory to first install the repo, then run the evaluation.
```
bash taskbench_eval.sh $INPUT $CONFIG $MODEL
# example below
bash taskbench_eval.sh sharegpt_data.json taskbench.yml meta-llama/Meta-Llama-3.1-8B-Instruct
```
`$INPUT` Path of the sharegpt taskbench file.
`$CONFIG` Path of the yaml config file for LLaMA-Factory.
`$MODEL` Model name or path.

TaskBench measures action and action input separately.  
*Action Evaluation*:
**Node F1**: Measures f1 of the predicted sequence of actions against the ground truth. (Order insensitive)  
**Edge F1**: Concatenate consecutive actions and compare against the ground truth. (Order sensitive)

*Action Input Evaluation*:
Assume action input is a json object with key-value pairs. Keys are the parameter names and values are the parameter values.
**Name F1**: Measures f1 of the predicted parameter names against the ground truth. (Order insensitive)  
**Value F1**: Measures f1 of the predicted parameter values against the ground truth. (Order insensitive)
