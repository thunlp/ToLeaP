import sys
import json
import argparse
from rouge import Rouge
sys.stdout.reconfigure(encoding='utf-8')


def get_answer(data):
    answer = data
    if answer.find("API-Request:") >= 0:
        answerAction = answer[answer.find("API Request:") + 15 : answer.find("(")-1]     
    return answerAction

def get_predict(preAnswer):
    if preAnswer.find("API Request:") >= 0:
        preAction = preAnswer[preAnswer.find("API Request:") + 13 : preAnswer.find("(")-1] 
        return preAction
    else:
        return "no_found"

def tool_accuarcy(evalData):
    global accuarcy
    
    for i in range(len(evalData)):
        try:
            answers = get_answer(evalData[i]["label"])
            preAction = get_predict(evalData[i]["predict"])
            if answers == preAction:
                accuarcy.append(i)
        except: continue
    print("True selection:", len(accuarcy), "All nums:", len(evalData), "Accuarcy:", len(accuarcy)/len(evalData)*100,"%")


def calculate_rouge_l_score(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score

def rouge_l(evalData):
    global accuarcy
    rouge = []
    for i in accuarcy:
        score = calculate_rouge_l_score(evalData[i]["label"], evalData[i]["predict"])
        rouge.append(score)
    if len(accuarcy) > 0:
        print ("Rouge of True selection:\n" + "Sum:",sum(rouge), " Average:", sum(rouge)/len(accuarcy),"\n")
    else:
        print ("No True tool selection, thus no rouge of true selection")
    rouge = []
    for i in range(len(evalData)):
        if len(evalData[i]["label"]) != 0 and len(evalData[i]["predict"]) != 0:  
            score = calculate_rouge_l_score(evalData[i]["label"], evalData[i]["predict"])
            rouge.append(score)
    print ("Rouge of All selection:\n" + "Sum:",sum(rouge), " Average:", sum(rouge)/len(evalData),"\n")
   

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="src/scripts/gp_sharegpt_format_lv3-train.jsonl")
    args = parser.parse_args()

    #Test_file follows generated_prediction format
    accuarcy = []
    f = open(args.test_file, encoding="utf-8")
    evalData = [json.loads(line) for line in f]
    max_len = len(evalData)
    tool_accuarcy(evalData)
    rouge_l(evalData)
    f.close
