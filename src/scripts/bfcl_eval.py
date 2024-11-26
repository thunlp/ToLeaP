from utils.llm import Message



from utils.llm import ...
call_llm = LLM("LLaMa-3.1-8b") # input prompt, output response

def input_to_target_benchmarkname(input):  % input_to_target_bfcl, input: unified schema
...
return target_input, target_output

def metric_cal_metricname_benchmarkname(pred, gt):
...
return metric

def evaluation(input, call_llm):
target_input, ground_truth = input_to_target_benchmarkname(input)
predict = call_llm(target_input) / predict = llamafactory_inference(target_input) ?
metric = metric_cal(predict, ground_truth)
return metric