from typing import Tuple
from utils.llm import LLM

class CriticModel(LLM):
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 40,
        gpu_memory_utilization: float = 0.9,
        use_api_model: bool = False,
        port= 8008
    ):
        super().__init__(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            use_api_model=use_api_model,
            port = port
        )
        
        self.system_prompt = """You are a critic agent. Evaluate if the response's reasoning process and final result fits the user query. 
Format your answer as:
Score: positive or negative
Analysis: explain why the response is effective or ineffective this should be in one paragraph not split into dot point"""

        self.eval_prompt = """Analyze this interaction:
                    human query: {query}
                    agent thought: {thought}
                    agent response: {response}
                    ground truths: {ground_truth}"""


    def _create_eval_messages(self, query: str, thought, response: str, ground_truth):
        eval_content = self.eval_prompt.format(
            query=query,
            thought=thought,
            response=response,
            ground_truth=ground_truth
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": eval_content}
        ]
        return messages

    def evaluate(self, query: str, thought,response: str, ground_truth):
        """评估回复质量"""
        try:
            messages = self._create_eval_messages(query, thought, response, ground_truth)
            eval_text = self._single_inference(
                messages=messages,
                temperature=0  
            )
            # 解析评估结果
            score = ""
            analysis = ""
            for line in eval_text.split('\n'):
                if line.startswith('Score:'):
                    score = line.replace('Score:', '').strip()
                elif line.startswith('Analysis:'):
                    analysis = line.replace('Analysis:', '').strip()
            
            return score, analysis
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return '', f"评估失败：{str(e)}"