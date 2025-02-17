AGENT_GENERAL_INSTRUCTION = "You will act as the {agent_name} agent and give helpful response to the latest user request. You have access to the entire conversation history, but you should focus on fulfilling the most recent request."

AGENT_ANSWER = "I have the final answer"

AGENT_ANSWER_PREFIX = "Now forget previous instructions on how to format your output. Based on your thoughts in previous conversations, please give your final answer according to the new instructions:\n\n"
AGENT_ANSWER_SYSTEM = "Background information:\n{background_info}\n"
AGENT_ANSWER_TOOLS = "Tools you can use:\n{tools}\n"

AGENT_MAP = {
    "Unix": "You are an expert in Unix command-line operations. Given a task (e.g., file management, process control, or system administration), provide precise terminal commands with brief explanations. ",
    "Economist": "You are an expert on economics. You can use economics tools to solve requests related to economics. ",
    "Critic": "You are an expert in giving feedback on the conversation between the user and the assistant. ",
    "Python": "You are an expert in Python programming. You can write Python code to solve related tasks. ",
    "Mathematics": "You are an expert in mathematics. You can use mathematical tools and formulae to solve requests related to mathematics. ",
    "History": "You are an expert in history. You can use historical tools and knowledge to solve requests related to history. ",
    # "Motivation": "Your role is to understand and clarify the user's underlying motivation or goal within the given context. You should identify the 'why' behind the user's request.",
    # "Propose": "Your role is to generate a proposal, request, or initial suggestion based on the identified motivation. This is the starting point for addressing the user's need.",
    # "Reasoning": "Your role is to engage in a reasoning process. This involves step-by-step thinking, planning, and problem-solving to achieve the stated goal.",
    # "Critic": "Your role is to act as an internal critic, evaluating the current state of the conversation or solution. You should identify errors, weaknesses, inconsistencies, or areas for improvement.",
    # "Analysis": "Your role is to perform a deep analysis of the current situation. This involves identifying past key information, assessing progress towards the overall goal and defining future objectives.",
    # "Summarize": "Your role is to provide concise summaries of the current state of the conversation, progress made, or key findings.",
    # "Pivot": "Your role is to recognize when the current strategy is failing or suboptimal. You should suggest a significant change in direction, a new approach, or an alternative strategy.",
    # "Output": "Your role is to generate the final output or response to the user's request. This should be a complete and well-formed answer, solution, or result. This is the terminal state.",
}

# Task specific instructions

TOOL_USING_FORMAT_INSTRUCTION = "You are trying to answer a question by calling tools. If output format is given below, please follow it to give the final answer. Otherwise, clearly state the name and parameters for each tool you are calling, separated by line breaks for each tool call. You should not include any other text in your response.\n" + '-' * 10 + "\n"

for k in AGENT_MAP:
    AGENT_MAP[k] = AGENT_MAP[k].format(agent_name=k)

class Agent:
    def __init__(self, name: str, llm, env_description: str = None):
        if name not in AGENT_MAP:
            raise ValueError(f"Invalid agent name: {name}. Possible values: {AGENT_MAP.keys()}")

        self._name = name
        self._system = AGENT_MAP[name] + "\n" + AGENT_GENERAL_INSTRUCTION.format(agent_name=name)
        if env_description:
            self._system += "\n\n" + env_description
        self._llm = llm

    def respond(self, history: list[dict]) -> str:
        messages = [
            {"role": "system", "content": self._system},
        ]
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        response = self._llm.single_generate_chat(messages)
        return response


