from utils.llm import LLM

class BaseModel(LLM):
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        use_api_model: bool = False,
        port= None
    ):
        super().__init__(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            use_api_model=use_api_model,
            port= port
        )

    def _generate(self, conversation_data, temperature: float = 0) -> str:
        try:
            messages = self._create_messages_from_sharegpt(conversation_data)
            response = self._single_inference(messages, temperature=temperature)
            return response.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return "" 
    
    def _batch_generate(self, conversations_batch, max_concurrent_calls=4, temperature=0.7):
        all_messages = [self._create_messages_from_sharegpt(case) for case in conversations_batch]
        try:
            return self._batch_inference(all_messages, max_concurrent_calls, temperature)
            
        except Exception as e:
            return ["" for _ in all_messages]