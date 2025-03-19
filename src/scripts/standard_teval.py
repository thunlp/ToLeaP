import mmengine

def format_percentage(value):
    return f"{value * 100:.2f}%"

def format_results(result_path):
    result = mmengine.load(result_path)
    
    instruct_json = (result['instruct_json']['json_format_metric'] + result['instruct_json']['json_args_em_metric']) / 2
    instruct_str = (result['instruct_json']['string_format_metric'] + result['instruct_json']['string_args_em_metric']) / 2
    
    formatted_results = {
        'Instruct': {
            'JSON': format_percentage(instruct_json),
            'String': format_percentage(instruct_str)
        },
        'Plan': {
            'String': format_percentage(result['plan_str']['f1_score']),
            'JSON': format_percentage(result['plan_json']['f1_score'])
        },
        'Reason': {
            'String': format_percentage(result['reason_str']['thought']),
            'JSON': format_percentage(result['rru_json']['thought'])
        },
        'Retrieve': {
            'String': format_percentage(result['retrieve_str']['name']),
            'JSON': format_percentage(result['rru_json']['name'])
        },
        'Understand': {
            'String': format_percentage(result['understand_str']['args']),
            'JSON': format_percentage(result['rru_json']['args'])
        },
        'Review': {
            'String': format_percentage(result['review_str']['review_quality']),
            'JSON': format_percentage(result['review_str']['review_quality'])
        }
    }
    
    return formatted_results

def print_results(formatted_results):
    """打印格式化后的结果"""
    for category, values in formatted_results.items():
        print(f"{category}:")
        print(f"  String: {values['String']}")
        print(f"  JSON: {values['JSON']}")
        print()

if __name__ == '__main__':
    import sys
    from mmengine import load
    
    if len(sys.argv) != 2:
        print("Usage: python format_results.py <result_path>")
        sys.exit(1)
        
    result_path = sys.argv[1]
    formatted = format_results(result_path)
    print_results(formatted) 