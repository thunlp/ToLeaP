from abc import ABC
from typing import List
from utils.llm import LLM
from cfg.config import Config

conf = Config()


class Agent(ABC):
    def __init__(self,
                 profile: str = None, 
                 base_llm: str = None, 
                 infer_method: str = None,
                 ):
        self.profile = profile
        self.base_llm = base_llm
        self.infer_method = infer_method


# Critic Agent is responsible for providing the high-quality feedback for reflection.
class CriticAgent(Agent):
    def __init__(self, 
                 profile: str = None,
                 base_llm: str = None,
                 infer_method: str = None,
                 gpu_memory_utilization: float = 0.9,
                 dtype: str = None, 
                 tensor_parallel_size: int = 1, 
                 use_api_model: bool = False, 
                 use_sharegpt_format: bool = False, 
                 max_past_message_include: int = -1,
                 temperature: int = 0,
                 ):
        super().__init__(profile, base_llm, infer_method)

        self.temperature = temperature

        if self.infer_method == "vllm":
            # TODO: infer through vllm
            self.llm = LLM(model=base_llm, 
                           gpu_memory_utilization=gpu_memory_utilization,
                           )
        elif self.infer_method == "hf":
            # TODO: infer through local hf model
            self.llm = LLM()
        elif self.infer_method == "openai":
            # TODO: infer through openai api
            self.llm = LLM()
        else:
            raise ValueError("Invalid Inference Method Name: Your input inference method name is not identified.")


    def feedback_gen(self, critic_context):
        # TODO: create message according to the profile (e.g., acting as a critic) and critic_context
        message = self.llm._create_messages_from_user()
        comment = self.llm._single_inference(messages=message, temperature=self.temperature)
        return comment
            

# EvalAgent is responsible for providing the true or false response to determine whether the task is finished.
class EvalAgent(Agent):
    def __init__(self, 
                 profile: str = None,
                 base_llm: str = None,
                 infer_method: str = None,
                 gpu_memory_utilization: float = 0.9,
                 dtype: str = None, 
                 tensor_parallel_size: int = 1, 
                 use_api_model: bool = False, 
                 use_sharegpt_format: bool = False, 
                 max_past_message_include: int = -1,
                 temperature: int = 0,
                 ):
        super().__init__(profile, base_llm, infer_method)

        self.temperature = temperature  # the temperature should be set to zero

        if self.infer_method == "vllm":
            # TODO: infer through vllm
            self.llm = LLM(model=base_llm, 
                           gpu_memory_utilization=gpu_memory_utilization,
                           )
        elif self.infer_method == "hf":
            # TODO: infer through local hf model
            self.llm = LLM()
        elif self.infer_method == "openai":
            # TODO: infer through openai api
            self.llm = LLM()
        else:
            raise ValueError("Invalid Inference Method Name: Your input inference method name is not identified.")


    def state_check(self, current_context):

        # TODO: create message according to the current state and agent profile.
        message = self.llm._create_messages_from_user()
        response = self.llm._single_inference()

        # TODO: complete and refine the response parsing rules
        if "Yes" in response:
            return True
        elif "No" in response:
            return False
        else:
            return False