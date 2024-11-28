import json
from argparse import ArgumentParser
from utils.bfcl import process_multiple_func_string

parser = ArgumentParser()
parser.add_argument('--result_file', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    data = [json.loads(line)['inference_log'] for line in open(args.result_file, 'r')]

    all_results = []

    for logs in data:
        conversations = []
        system = None
        for i, log in enumerate(logs):
            keys = sorted(log.keys())
            assistant = None
            observations = []
            for k in keys:
                for msg in log[k]:
                    if msg['role'] == 'system':
                        system = msg['content']
                    elif msg['role'] == 'user':
                        conversations.append({
                            'from': 'human',
                            'value': msg['content']
                        })
                    elif msg['role'] == 'assistant':
                        assistant = msg['content']
                    elif msg['role'] == 'handler_log':
                        if msg['content'] == 'Successfully decoded model response.':
                            function_calls = [process_multiple_func_string(func) for func in msg['model_response_decoded']]
                            conversations.append({
                                'from': 'function_call',
                                'value': json.dumps(function_calls)
                            })
                    elif msg['role'] == 'tool':
                        observations.append(msg['content'])

            if assistant is not None:
                if len(observations) > 0:
                    conversations.append({
                        'from': 'observation',
                        'value': json.dumps(observations)
                    })
                    conversations.append({
                        'from': 'gpt',
                        'value': "Executed function calls: " + assistant
                    })
                else:
                    conversations.append({
                        'from': 'gpt',
                        'value': "No function calls executed. " + assistant
                    })

        all_results.append({
            'conversations': conversations,
            'system': system
        })

    with open(args.result_file + '_sharegpt.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    
    