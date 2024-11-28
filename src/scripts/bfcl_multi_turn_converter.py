import json
import os
from argparse import ArgumentParser
from utils.bfcl import process_multiple_func_string

parser = ArgumentParser()
parser.add_argument('--model', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    result_dir = args.model.replace('/', '_')
    
    # Process all json files in result_dir
    for result_file in os.listdir(result_dir):
        if not result_file.endswith('.json'):
            continue
            
        result_path = os.path.join(result_dir, result_file)
        data = [json.loads(line).get('inference_log', []) for line in open(result_path, 'r')]

        all_results = []

        for logs in data:
            conversations = []
            system = None
            for i, log in enumerate(logs):
                keys = sorted(log.keys())
                keys = [keys[0]] + sorted(keys[1:], key=lambda x: int(x.split('_')[1]))
                assistant = None
                for k in keys:
                    observations = []
                    step_completed = False
                    for msg in log[k]:
                        if msg['role'] == 'system':
                            system = msg['content']
                        elif msg['role'] == 'user':
                            conversations.append({
                                'from': 'human',
                                'value': msg['content']
                            })
                        elif msg['role'] == 'handler_log':
                            if msg['content'] == 'Successfully decoded model response.':
                                function_calls = [process_multiple_func_string(func) for func in msg['model_response_decoded']]
                                conversations.append({
                                    'from': 'function_call',
                                    'value': json.dumps(function_calls)
                                })
                            elif msg['content'] == 'Error decoding the model response. Proceed to next turn.':
                                step_completed = True
                                conversations.append({
                                    'from': 'gpt',
                                    'value': "Current turn completed. Moving on to the next turn."
                                })
                        elif msg['role'] == 'tool':
                            observations.append(msg['content'])

                    if len(observations) > 0:
                        conversations.append({
                            'from': 'observation',
                            'value': json.dumps(observations)
                        })

            all_results.append({
                'conversations': conversations,
                'system': system
            })

        output_path = os.path.join(result_dir, result_file.replace('.json', '_sharegpt.json'))
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)