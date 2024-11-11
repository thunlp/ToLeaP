TOOLALPACA_SYSTEM = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions with the help of some tools.
You have access to the following tools:

{tools}

The chat follows this format:
USER: the user's question
ASSISTANT Thought: the assistant's inner thought about what to do next 
ASSISTANT Action: the action to take, must be one of [getDetails, sendHttpRequest, getClientRequestData, testProxyHeaders, simulateStatusCode].
ASSISTANT Action Input: the input for the action, in JSON format.
ASSISTANT Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
ASSISTANT Thought: summarize the information gathered
ASSISTANT Response: the final response to the user
USER: user's next question
...

Begin!

USER:{instruction}
ASSISTANT Thought: 
"""

TOOLALPACA_EVAL = """Given the documentation of a REST API and a task instruction, I need you to evaluate whether the solution provided by my AI assistant aligns with the standard answer. 

Follow these guidelines:
1. You need to assess both the process and final response of the AI assistant's solution.
2. For the process, refer to the standard answer:
 - The standard answer only includes function names and parameters, while the AI assistant's solution also includes function returns. Therefore, it is acceptable to adjust the call situation based on the function return, such as retrying when the function errors, calling function `getDetails` for more information, calling function `retrievalDataFromFile` when function's return is too long.
 - Random calls to unrelated functions are not allowed.
 - The solution must contain all the steps in the standard answer.
 - The necessary parameters of the function need to be consistent with the standard answer. Parameters not mentioned in the instruction can be inconsistent.
3. You need to comprehensively judge whether the final response of the solution accurately summarizes the entire call process and provides a reasonable response to the initial instruction.
4. You need to first analyze the entire solution according to the guidelines, then give your answer. Your output should adhere to the format: 

## Documentation
${documentation}

## Task Instruction
${instruction}

## Standard Answer
${standard}

## AI Assistant's Solution
${solution}

## Analysis 
{some analysis}

## Results 
Process Correctness: one of [Yes, No, Uncertain] 
Final Response Correctness: one of [Yes, No, Uncertain]
"""