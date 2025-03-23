#!/usr/bin/env python3

import argparse
import json
import jsonlines
import warnings
import re
import string

def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")

def split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)

def normalize_str(input_str, remove_punct=True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()

def question_scorer(model_answer: str, ground_truth: str) -> bool:
    def is_float(x):
        try: float(x); return True
        except: return False

    if model_answer is None: model_answer = "None"

    if is_float(ground_truth):
        return normalize_number_str(model_answer) == float(ground_truth)
    if any(c in ground_truth for c in [",", ";"]):
        gt_elems, ma_elems = split_string(ground_truth), split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            warnings.warn("List length mismatch → False")
            return False
        for ma, gt in zip(ma_elems, gt_elems):
            if is_float(gt):
                if normalize_number_str(ma) != float(gt): return False
            else:
                if normalize_str(ma, False) != normalize_str(gt, False): return False
        return True
    return normalize_str(model_answer) == normalize_str(ground_truth)

def load_inference(path):
    with open(path, encoding="utf-8") as f:
        return {e["Question"]: {"response": e.get("response",""), "file_name": e.get("file_name","")} for e in json.load(f)}

def load_ground_truth(path):
    out={}
    with jsonlines.open(path) as reader:
        for obj in reader:
            out[obj["Question"]] = obj["Final answer"]
    return out

def main(pred_path, gt_path):
    preds, truths = load_inference(pred_path), load_ground_truth(gt_path)
    all_results, corrects, incorrects = [], [], []

    for q, gt in truths.items():
        entry = preds.get(q, {"response":"", "file_name":""})
        pred, fname = entry["response"], entry["file_name"]
        correct = question_scorer(pred, gt)
        record = {"Question":q, "file_name":fname, "Prediction":pred, "GroundTruth":gt, "Correct":correct}
        all_results.append(record)
        (corrects if correct else incorrects).append(record)

    accuracy = sum(r["Correct"] for r in all_results) / len(all_results)
    print(f"Evaluated {len(all_results)} examples — Accuracy: {accuracy:.2%}")

    for name, data in [("evaluation_results.json", all_results),
                       ("correct_cases.json", corrects),
                       ("incorrect_cases.json", incorrects)]:
        with open(name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    args = parser.parse_args()
    main(args.predictions, args.ground_truth)
