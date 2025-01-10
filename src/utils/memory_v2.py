from typing import Dict, List, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)

class MemoryV2:
    def __init__(self, save_path: str = "memory_v2.json"):
        """
        初始化内存系统
        
        存储格式:
        {
            "conversation_id": {
                "turns": [
                    {
                        "turn_id": 1,
                        "query": str,
                        "gt_history": [],  # 存储已确认的对话历史
                        "iterations": {  # 改为dict, key为外部iteration_id
                            "1": [  # 每个iteration包含多个retry尝试
                                {
                                    "thought": str,
                                    "response": str,
                                    "ground_truth": str,
                                    "feedback": str,
                                    "score": int,
                                    "retry_count": int
                                }
                            ],
                            "2": [
                                {...}
                            ]
                        }
                    }
                ]
            }
        }
        """
        self.memory: Dict = {}
        self.save_path = save_path
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                logger.info(f"已从 {self.save_path} 加载 {len(self.memory)} 个对话记录")
            except Exception as e:
                logger.error(f"加载内存文件失败: {e}")
                self.memory = {}

    def save_to_disk(self):
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(self.memory)} 个对话记录到 {self.save_path}")
        except Exception as e:
            logger.error(f"保存内存文件失败: {e}")

    def save_turn(self, 
                 conv_id: str, 
                 turn_id: int, 
                 query: str, 
                 gt_history: list, 
                 thought: str, 
                 response: str, 
                 ground_truth: str, 
                 feedback: str, 
                 score: int,
                 iteration: int,
                 retry_count: int = 0):
        """保存一轮对话的迭代结果"""
        if conv_id not in self.memory:
            self.memory[conv_id] = {"turns": []}
        
        # 查找或创建turn
        turns = self.memory[conv_id]["turns"]
        turn = None
        for t in turns:
            if t["turn_id"] == turn_id:
                turn = t
                break
        
        if turn is None:
            turn = {
                "turn_id": turn_id,
                "query": query,
                "gt_history": gt_history,
                "iterations": {}  # 改为dict
            }
            turns.append(turn)
        
        # 确保iteration存在
        if str(iteration) not in turn["iterations"]:
            turn["iterations"][str(iteration)] = []
        
        # 添加新的尝试结果
        turn["iterations"][str(iteration)].append({
            "thought": thought,
            "response": response,
            "ground_truth": ground_truth,
            "feedback": feedback,
            "score": score,
            "retry_count": retry_count
        })
        
        # 更新gt_history
        if gt_history and gt_history != turn["gt_history"]:
            turn["gt_history"] = gt_history
        
        self.save_to_disk()

    def get_turn_history(self, conv_id: str, turn_id: int) -> Optional[dict]:
        """获取特定对话轮次的所有历史记录"""
        if conv_id not in self.memory:
            return None
        
        for turn in self.memory[conv_id]["turns"]:
            if turn["turn_id"] == turn_id:
                return turn
        return None

    def get_last_iteration_result(self, conv_id: str, turn_id: int, iteration: int) -> Optional[dict]:
        """获取上一轮iteration的最后一次尝试结果"""
        turn = self.get_turn_history(conv_id, turn_id)
        if not turn or str(iteration-1) not in turn["iterations"]:
            return None
        
        last_tries = turn["iterations"][str(iteration-1)]
        if not last_tries:
            return None
            
        return last_tries[-1]  # 返回最后一次尝试

    def get_best_response(self, conv_id: str, turn_id: int) -> Optional[dict]:
        """获取特定轮次中得分最高的响应（跨所有iterations）"""
        turn = self.get_turn_history(conv_id, turn_id)
        if not turn:
            return None
            
        best_response = None
        best_score = float('-inf')
        
        for iteration_tries in turn["iterations"].values():
            for try_result in iteration_tries:
                score = try_result["score"]
                retry_count = try_result.get("retry_count", 0)
                
                if score > best_score or (score == best_score and retry_count < best_response.get("retry_count", 0)):
                    best_score = score
                    best_response = try_result
        
        return best_response if best_response else None

    def get_conversation_history(self, conv_id: str) -> Optional[List[dict]]:
        """获取整个对话的历史记录"""
        if conv_id not in self.memory:
            return None
        return self.memory[conv_id]["turns"]

    def get_statistics(self) -> Dict:
        """获取内存统计信息"""
        total_conversations = len(self.memory)
        total_turns = 0
        total_iterations = 0
        total_retries = 0
        positive_scores = 0
        
        for conv in self.memory.values():
            for turn in conv["turns"]:
                total_turns += 1
                for iteration_tries in turn["iterations"].values():
                    total_iterations += len(iteration_tries)
                    for try_result in iteration_tries:
                        total_retries += try_result.get("retry_count", 0)
                        if try_result["score"] > 0:
                            positive_scores += 1
        
        return {
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "total_iterations": total_iterations,
            "total_retries": total_retries,
            "positive_ratio": positive_scores / total_iterations if total_iterations > 0 else 0,
            "avg_retries": total_retries / total_iterations if total_iterations > 0 else 0
        } 

    def get_previous_turns_history(self, conv_id: str, current_turn_id: int) -> List[Dict]:
        """获取当前turn之前的所有成功对话历史"""
        history = []
        if conv_id not in self.memory:
            return history
        
        for turn in self.memory[conv_id]["turns"]:
            if turn["turn_id"] < current_turn_id:  # 只获取之前的turn
                # 遍历每个iteration
                for iter_id, iter_tries in turn["iterations"].items():
                    # 找到这个iteration中最后一个positive的结果
                    for try_data in reversed(iter_tries):
                        if try_data["score"] == 1:  # score是数字
                            history.extend([
                                {"from": "human", "value": turn["query"]},
                                {"from": "assistant", "value": try_data["response"]}
                            ])
                            break  # 找到positive就跳出内层循环
                    break  # 只看最新的iteration
        return history 

    def convert_to_sharegpt(self, memory_path: str, testdata_path: str, output_path: str, thought_prompt=''):
        """转换memory数据为ShareGPT格式
        
        Args:
            memory_path: memory json文件路径
            testdata_path: testdata.json文件路径,用于获取system prompt
            output_path: 输出文件路径
        """
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        with open(testdata_path, 'r', encoding='utf-8') as f:
            testdata = json.load(f)
        
        output_data = []
        

        for idx, conv_data in memory_data.items():
            idx = int(idx)  - 1  # 转换为数字以匹配testdata的索引
            conv_output = {"conversations": []}
            
            # 从testdata获取对应idx的system prompt
            if idx >= 0 and idx < len(testdata):
                # conv_output["system"] = testdata[idx - 1 ].get("system", "") + thought
                conv_output["system"] = testdata[idx].get("system", "") + thought_prompt
            
            # 遍历turns直到遇到negative结果
            turns = conv_data["turns"]
            for turn in turns:
                # 获取最后一个iteration
                last_iter_id = str(max(int(i) for i in turn["iterations"].keys()))
                last_iter = turn["iterations"][last_iter_id][-1]  # 取最后一次尝试
                
                # 如果最后一个iteration是negative，停止处理这个对话
                if last_iter["score"] == -1:
                    break
                
                # 添加human turn
                conv_output["conversations"].append({
                    "from": "human",
                    "value": turn["query"]
                })
                
                thought = last_iter['thought']
                if thought.lower().startswith("thought:"):
                    thought = thought[8:].strip() 

                # 构建gpt回复
                gpt_response = (
                    f"<thought> {thought} </thought> \n "
                    f"<output> {last_iter['ground_truth']} </output>"
                )
                
                conv_output["conversations"].append({
                    "from": "gpt",
                    "value": gpt_response
                })
            
            # 只保存有对话的数据
            if len(conv_output["conversations"]) > 0:
                output_data.append(conv_output)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Converted {len(output_data)} conversations to {output_path}") 