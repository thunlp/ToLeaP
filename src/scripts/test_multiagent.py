import json
import argparse
from utils.llm import LLM
from utils.agent import *
from utils.memory_agent import MultiAgentMemory
import os
from typing import List

def illustrate_messages(messages):
    for m in messages:
        print(m['role'] + ': ' + m['content'] + '\n')
        print('-' * 10)

def create_agents(llm):
    agents = {}
    for k in AGENT_MAP:
        agents[k] = Agent(k, llm)
    return agents

def construct_system_prompt(system_prefix):
    system_prompt = system_prefix + "\n\n"
    system_prompt += "When handling complex reasoning tasks, you have access to the following agents. Use these agents to help you progress towards the final answer. You have access to the following agents:\n\nAgents:\n"
    for agent_name, agent_description in AGENT_MAP.items():
        system_prompt += f'{agent_name}: {agent_description.replace("You are", "This agent is")}\n'
    system_prompt += "\n"
    system_prompt += "You should always activate exactly one agent until you are ready to give the final answer. Your response when activating an agent should consist of exactly two lines in the format:\n"
    system_prompt += f"""Thought: [Your step-by-step reasoning on how to solve the task]
Activate Agent: [Name of the agent to activate]

When you are ready to give the final answer, give the answer directly without any other text. Some additional reminders:
- You can only activate one agent at a time
- Maximum 10 agent calls before finalizing the answer
- 'Activate Agent' should only be followed by the name of the agent to activate without any other text."""

    return system_prompt

def main():
    parser = argparse.ArgumentParser(description='Run multi-agent system')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                      help='Model name or path')
    parser.add_argument('--data_path', type=str, default='sft_bfcl_multi_turn_base.json',
                      help='Path to input data file')
    parser.add_argument('--exp', type=str, default='bfcl_multi_turn',
                      help='Experiment name')
    parser.add_argument('--is_api', type=bool, default=False,
                      help='Whether to use API')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                      help='Tensor parallel size')
    parser.add_argument('--batch_size', type=int, default=20,
                      help='Batch size for inference')
    parser.add_argument('--max_input_tokens', type=int, default=32000,
                      help='Maximum input tokens')
    parser.add_argument('--max_output_tokens', type=int, default=4096,
                      help='Maximum output tokens')

    args = parser.parse_args()

    input_data = json.load(open(args.data_path, "r"))
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, 
              batch_size=args.batch_size, max_input_tokens=args.max_input_tokens, 
              max_output_tokens=args.max_output_tokens)

    all_answers = []

    memory = MultiAgentMemory(save_path="multiagent_memory_{}.json".format(args.exp))
    
    for i, d in enumerate(input_data[:50]):
        print('Starting turn ' + str(i + 1))
        all_messages = []
        agent_messages = []
        
        # Construct system instruction once per conversation
        system_prefix = ""
        if 'system' in d:
            system_prefix = d['system'] + "\n\n"
        system_prefix += '-' * 10 + "\n"
        system_instruction = construct_system_prompt(system_prefix)
        
        # Start conversation with system instruction
        conv_idx = memory.start_conversation(system_instruction)
        agents = create_agents(llm)
        all_answers.append([])

        # Add system instruction to messages once
        all_messages.append({"role": "system", "content": system_instruction})
        
        for i in range(0, len(d['conversations']), 2):
            print('Handling question ' + str(i // 2 + 1))
            user_prompt = d['conversations'][i]['value']
            label = d['conversations'][i + 1]['value']

            # Remove system_instruction parameter
            turn_idx = memory.add_turn(conv_idx, user_prompt, label)
            
            all_messages.append({"role": "user", "content": user_prompt})
            max_agent_calls = 10
            agent_calls = 0
            
            while True:
                response = llm.single_generate_chat(all_messages)
                if "Thought:" not in response and "Activate Agent:" not in response:
                    final_answer = response
                    all_messages.append({"role": "assistant", "content": final_answer})
                    memory.set_final_answer(conv_idx, turn_idx, final_answer)
                    all_answers[-1].append(final_answer)
                    break
                elif agent_calls >= max_agent_calls:
                    # Generate final answer after max agent calls
                    final_prompt = {"role": "user", "content": "Based on all the agent interactions above, please provide a final answer to the original question. Format your response as 'Final Answer: <your answer>'"}
                    all_messages.append(final_prompt)
                    final_response = llm.single_generate_chat(all_messages)
                    
                    if "Final Answer:" in final_response:
                        final_answer = final_response.split("Final Answer:")[1].strip()
                    else:
                        final_answer = "Error: Unable to generate final answer after max agent calls"
                        
                    memory.set_final_answer(conv_idx, turn_idx, final_answer)
                    all_answers[-1].append(final_answer)
                    print('Max agent calls reached - generated final answer')
                    break
                else:
                    all_messages.append({"role": "assistant", "content": response})
                    print('Calling Agent')
                    agent_calls += 1
                
                try:
                    assert len(response.split("\n")) == 2, "Response must be exactly 2 lines"
                    thought = response.split("\n")[0].split("Thought: ")[1].strip()
                    agent_to_call = response.split("\n")[1].split("Activate Agent: ")[1].strip()
                    assert agent_to_call in AGENT_MAP, f"Agent {agent_to_call} not in AGENT_MAP"
                    
                    # Record base model's thought before agent interaction
                    memory.add_base_model_thought(conv_idx, turn_idx, thought, agent_to_call)
                    
                    agent_messages.append({"role": "user", "content": thought})
                except Exception as e:
                    # print('The response ###{}### contains error'.format(response))
                    # Record the raw response when parsing fails
                    memory.add_base_model_error_response(conv_idx, turn_idx, response)
                    all_messages.append({"role": "user", "content": f"Error: {e}\nPlease follow the specified agent calling format and try again."})
                    continue

                # Agent response code moved outside try block
                agent_response = agents[agent_to_call].respond(agent_messages)
                agent_messages.append({"role": "assistant", "content": agent_response})
                memory.add_agent_interaction(conv_idx, turn_idx, agent_to_call, thought, agent_response)
                print('Agent Called')
                all_messages.append({"role": "user", "content": f"AGENT {agent_to_call}: {agent_response}"})

        # break

if __name__ == "__main__":
    main()