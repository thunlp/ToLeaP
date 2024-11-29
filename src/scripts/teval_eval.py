import json
import re
import os

class FunctionCallEvaluator:
    def __init__(self):
        self.action_pattern = r'Action:\s*([^\n]*)'
        self.action_input_pattern = r'Action Input:\s*({[^}]*})'
        
    def extract_action_info(self, text):
        try:
            if text.strip().startswith("{"):
                data = json.loads(text)
                action = data.get("Action") or data.get("action") or data.get("Name") or data.get("name")
                action_input = data.get("Action_Input") or data.get("args") or data.get("Args")
                return action, action_input
            
         
            action_match = re.search(self.action_pattern, text)
            action = action_match.group(1).strip() if action_match else None

            input_match = re.search(self.action_input_pattern, text)
            action_input = json.loads(input_match.group(1)) if input_match else None

            return action, action_input
        except Exception as e:
            return None, None
    
    def compute_args_em_metric(self, gt_action, pred_action, gt_args, pred_args):
        cnt = 0.
        if gt_action == pred_action:
            cnt += 1.
        num_args = len(gt_args) + 1 
        for gt_key in gt_args:
            pred_val = pred_args.get(gt_key, "")
            if pred_val == gt_args[gt_key]:
                cnt += 1.
        return cnt / num_args
    
    def evaluate_single(self, example):
        try:
            label = json.loads(example["label"])
            gt_name = label["name"]
            gt_args = label["arguments"]

            predict = example["predict"]
            pred_action, pred_args = self.extract_action_info(predict)

            if pred_action is None or pred_args is None:
                return {
                    "format_score": 0,
                    "args_score": 0,
                    "total_score": 0,
                    "error": "Invalid format or missing fields"
                }

            format_score = 1

            args_score = self.compute_args_em_metric(gt_name, pred_action, gt_args, pred_args)

            total_score = (format_score + args_score) / 2

            return {
                "format_score": format_score,
                "args_score": args_score,
                "total_score": total_score,
                "error": None
            }
        except Exception as e:
            return {
                "format_score": 0,
                "args_score": 0,
                "total_score": 0,
                "error": str(e)
            }
    
    def evaluate_dataset(self, examples):
        results = []
        total_format_score = 0
        total_args_score = 0
        total_score = 0

        for example in examples:
            result = self.evaluate_single(example)
            results.append(result)

            total_format_score += result["format_score"]
            total_args_score += result["args_score"]
            total_score += result["total_score"]


        n = len(examples)
        return {
            "results": results,
            "summary": {
                "avg_format_score": total_format_score / n,
                "avg_args_score": total_args_score / n,
                "avg_total_score": total_score / n
            }
        }

if __name__ == "__main__":
    import json
    import os

    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'gp_sft_teval_ins.jsonl')

    evaluator = FunctionCallEvaluator()

    try:
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): 
                    dataset.append(json.loads(line))

        dataset_results = evaluator.evaluate_dataset(dataset)

        output_path = os.path.join(base_path, 'teval_evaluation_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, ensure_ascii=False, indent=2)

        print("\nEvaluation Summary:")
        print(f"Number of examples evaluated: {len(dataset)}")
        print(f"Average format score: {dataset_results['summary']['avg_format_score']:.3f}")
        print(f"Average args score: {dataset_results['summary']['avg_args_score']:.3f}")
        print(f"Average total score: {dataset_results['summary']['avg_total_score']:.3f}")
        print(f"\nDetailed results saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON line: {str(e)}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
