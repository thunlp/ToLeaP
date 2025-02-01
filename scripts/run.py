import os
import click
import yaml
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import mmengine
from models.base_model import BaseModel
from models.critic_model import CriticModel
from utils.memory import Memory
from utils.memory_v2 import MemoryV2
from utils.data_loader import load_dataset

logger = logging.getLogger(__name__)



'''
main:
1. single_iteration(dataset, base_model, critic_model, memory, iteration, batch_size)
    - 输入: 数据集、两个模型实例、memory、当前迭代次数、batch_size
    - 处理流程：
        a. 批量处理数据集中的对话
        b. 对每个human query
            - 从memory获取历史生成记录(如果有)
            - 将历史最佳回复作为提示(第2轮开始)
            - 生成新回复并评估
            - 更新memory
        - 返回：当前轮次的所有生成结果 (full dataset)

2. main(config, iterations)
        a. 加载配置和初始化（模型、内存、日志等）
        b. 处理每个输入文件：
            - 执行多轮迭代(默认5轮)
            - 保存每轮结果
            - 生成最终微调数据

3. create_tuning_data(dataset, memory, final_iteration)
    - 生成最终微调数据集
    - 处理：
        - 遍历原始数据集
        - 为每个query选择得分最高的回复
        - 转换为微调格式
'''
thought_prompt = """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer.concise final answer based on the comprehensive analysis.
                        Remeber to format your answer as:
                        Thought: your step by step reasoning into one paragraph only
                        Output: the final output you going to provide to the user query"""


def process_conversation(
    conv: Dict,
    base_model: BaseModel,
    critic_model: CriticModel,
    memory: MemoryV2,
    iteration: int,
    conv_id: str,
    max_retry: int = 2
) -> List[Dict]:
    """处理单个对话的所有轮次"""
    results = []
    
    # 处理每一轮对话
    for turn_idx in range(0, len(conv["conversations"]), 2):
        if turn_idx + 1 >= len(conv["conversations"]):
            break
            
        human_turn = conv["conversations"][turn_idx]
        gpt_turn = conv["conversations"][turn_idx + 1]
        
        if human_turn["from"] != "human" or gpt_turn["from"] != "gpt":
            continue
            
        turn_id = (turn_idx // 2) + 1
        query = human_turn["value"]
        ground_truth = gpt_turn["value"]
        
        # 获取之前turn的历史
        gt_history = memory.get_previous_turns_history(conv_id, turn_id)
        
        # 尝试生成回复，最多retry次
        retry_count = 0
        best_result = None
        last_result = None
        
        while retry_count < max_retry:
            try:
                # 准备prompt
                if retry_count == 0:
                    prompt = {
                        "system": f"{conv.get('system', '')}\n\n{thought_prompt}",
                        "conversations": gt_history + [{"from": "human", "value": query}]
                    }
                else:
                    prompt = {
                        "system": f"""{conv.get('system', '')}\n\n{thought_prompt}
Previous attempt failed. Here's the feedback:
{feedback}
Previous thought process:
{thought}
Previous response:
{final_response}
Please improve your response by addressing the feedback.""",
                        "conversations": gt_history + [{"from": "human", "value": query}]
                    }
                
                # 生成回复
                response = base_model._generate(prompt, temperature=0)
                thought, final_response = extract_thought_and_response(response)
                
                # 评估回复
                score, feedback = critic_model.evaluate(query, thought, final_response, ground_truth)
                
                # 保存到memory
                memory.save_turn(
                    conv_id=conv_id,
                    turn_id=turn_id,
                    query=query,
                    gt_history=gt_history,
                    thought=thought,
                    response=final_response,
                    ground_truth=ground_truth,
                    feedback=feedback,
                    score=1 if score == "Positive" else -1,
                    iteration=iteration,
                    retry_count=retry_count
                )
                
                # 构建结果
                last_result = {
                    "conversations": [
                        {"from": "human", "value": query},
                        {
                            "from": "gpt",
                            "value": final_response,
                            "metadata": {
                                "iteration": iteration,
                                "score": score,
                                "feedback": feedback,
                                "thought": thought,
                                "turn_id": turn_id,
                                "retry_count": retry_count
                            }
                        }
                    ],
                    "conv_id": conv_id,
                    "turn_id": turn_id
                }
                
                if score == "Positive":
                    best_result = last_result
                    break
                
            except Exception as e:
                logger.error(f"处理错误 (query: {query[:50]}...): {e}")
            
            retry_count += 1
        
        # 添加结果并检查是否继续
        if not best_result:
            results.append(last_result)  # 添加最后一次尝试的结果
            logger.warning(f"对话 {conv_id} 在轮次 {turn_id} 未能生成满意回复，终止对话")
            break  # 终止对话
        else:
            results.append(best_result)  # 添加positive结果
    
    return results

def single_iteration(
    dataset: List[Dict],
    base_model: BaseModel,
    critic_model: CriticModel,
    memory: MemoryV2,
    iteration: int,
    batch_size: int = 2  # 改为默认2个conversation一批
) -> List[Dict]:
    """单次迭代处理"""
    logger.info(f"开始第 {iteration} 轮迭代...")
    all_results = []
    
    # 按batch处理对话
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_results = []
        
        logger.info(f"处理对话批次 {batch_start//batch_size + 1}, conversations {batch_start+1} to {batch_end}")
        
        # 处理这个batch的对话
        for conv_idx in range(batch_start, batch_end):
            conv = dataset[conv_idx]
            conv_id = str(conv_idx + 1)
            
            if iteration == 1:
                results = process_conversation(
                    conv=conv,
                    base_model=base_model,
                    critic_model=critic_model,
                    memory=memory,
                    iteration=iteration,
                    conv_id=conv_id
                )
            else:
                results = process_iteration(
                    conv=conv,
                    base_model=base_model,
                    critic_model=critic_model,
                    memory=memory,
                    iteration=iteration,
                    conv_id=conv_id
                )
            
            batch_results.extend(results)
        
        # 每处理完一个batch就保存memory
        memory.save_to_disk()
    
        
        all_results.extend(batch_results)
    
    return all_results

def process_iteration(
    conv: Dict,
    base_model: BaseModel,
    critic_model: CriticModel,
    memory: MemoryV2,
    iteration: int,
    conv_id: str
) -> List[Dict]:
    """处理后续迭代轮次"""
    results = []
    
    for turn_idx in range(0, len(conv["conversations"]), 2):
        if turn_idx + 1 >= len(conv["conversations"]):
            break
            
        human_turn = conv["conversations"][turn_idx]
        gpt_turn = conv["conversations"][turn_idx + 1]
        
        if human_turn["from"] != "human" or gpt_turn["from"] != "gpt":
            continue
            
        turn_id = (turn_idx // 2) + 1
        query = human_turn["value"]
        ground_truth = gpt_turn["value"]
        
        # 获取之前turn的历史
        gt_history = memory.get_previous_turns_history(conv_id, turn_id)
        
        # 获取上一轮的结果
        last_result = memory.get_last_iteration_result(conv_id, turn_id, iteration)
        if last_result:
            # 准备prompt
            prompt = {
                "system": f"""{conv.get('system', '')}\n\n{thought_prompt}
Previous iteration's response (score: {last_result['score']}):
{last_result['response']}
Previous thought process:
{last_result['thought']}
Please provide an improved response.""",
                "conversations": gt_history + [{"from": "human", "value": query}]
            }
            
            try:
                # 生成新回复
                response = base_model._generate(prompt, temperature=0)
                thought, final_response = extract_thought_and_response(response)
                
                # 评估回复
                score, feedback = critic_model.evaluate(query, thought, final_response, ground_truth)
                
                # 保存到memory
                memory.save_turn(
                    conv_id=conv_id,
                    turn_id=turn_id,
                    query=query,
                    gt_history=gt_history,
                    thought=thought,
                    response=final_response,
                    ground_truth=ground_truth,
                    feedback=feedback,
                    score=1 if score == "Positive" else -1,
                    iteration=iteration,
                    retry_count=0
                )
                
                # 保存结果
                results.append({
                    "conversations": [
                        {"from": "human", "value": query},
                        {
                            "from": "gpt",
                            "value": final_response,
                            "metadata": {
                                "iteration": iteration,
                                "score": score,
                                "feedback": feedback,
                                "thought": thought,
                                "turn_id": turn_id
                            }
                        }
                    ],
                    "conv_id": conv_id,
                    "turn_id": turn_id
                })
                
            except Exception as e:
                logger.error(f"处理错误 (query: {query[:50]}...): {e}")
    
    return results

def extract_thought_and_response(combined_text: str) -> Tuple[str, str]:
    """从模型输出中分离thought和最终response"""
    if "output:" in combined_text.lower():
        parts = combined_text.lower().split("output:")
        thought = parts[0].strip()
        if thought.lower().startswith("thought:"):
            thought = thought.strip()
        response = parts[1].strip()
        return thought, response

    separators = ["response:", "final answer:", "answer:"]
    for separator in separators:
        if separator in combined_text.lower():
            parts = combined_text.lower().split(separator)
            thought = parts[0].strip()
            if thought.lower().startswith("thought:"):
                thought = thought[7:].strip()
            response = parts[1].strip()
            return thought, response
    
    paragraphs = [p.strip() for p in combined_text.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        thought = '\n\n'.join(paragraphs[:-1])
        response = paragraphs[-1]

        return thought, response
    
    # 如果无法分离，返回原始文本作为response
    return "", combined_text.strip()

def check_termination(dataset: List[Dict], memory: MemoryV2, iteration: int) -> bool:
    """
    检查是否满足终止条件
    
    终止条件：
    1. 不是第一轮 (iteration > 1)
    2. 75% positive rate
    
    Returns:
        bool: 是否应该终止迭代
    """
    if iteration <= 1:
        return False
        
    total_positive = 0
    total_conversations = len(dataset)
    
    for conv_idx in range(total_conversations):
        conv_id = str(conv_idx + 1)
        turn_id = 1  # 当前只处理第一轮对话
        
        # 获取最佳响应
        best_response = memory.get_best_response(conv_id, turn_id)
        if best_response and best_response['score'] > 0:
            total_positive += 1
    
    quality_ratio = total_positive / total_conversations
    threshold_met = quality_ratio >= 0.75
    
    logger.info(f"质量检查 - 轮次 {iteration}:")
    logger.info(f"- 总对话数: {total_conversations}")
    logger.info(f"- 正面评价数: {total_positive}")
    logger.info(f"- 质量比例: {quality_ratio:.2%}")
    logger.info(f"- 是否达到终止条件: {threshold_met}")
    
    return threshold_met

@click.command()
@click.option("--config", type=str, default="config/config.yaml", help="配置文件路径")
@click.option("--max_iterations", type=int, default=20, help="最大迭代次数")
def main(config: str, max_iterations: int):
    # 加载配置
    cfg = load_config(config)
    
    # 创建输出目录
    output_dir = cfg["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化日志
    setup_logging(output_dir)
    
    try:
        base_model = BaseModel(
            model=cfg["model"]["base_model"]["path"],
            tensor_parallel_size=cfg["model"]["base_model"]["tensor_parallel_size"],
            gpu_memory_utilization=cfg["model"]["base_model"]["gpu_memory_utilization"],
            use_api_model=cfg["model"]["base_model"]["use_api"],
            port=cfg["model"]["base_model"]["server"]["port"]
        )
        
        critic_model = CriticModel(
            model=cfg["model"]["critic_model"]["path"],
            tensor_parallel_size=cfg["model"]["critic_model"]["tensor_parallel_size"],
            gpu_memory_utilization=cfg["model"]["critic_model"]["gpu_memory_utilization"],
            use_api_model=cfg["model"]["critic_model"]["use_api"],
            port=cfg["model"]["critic_model"]["server"]["port"]
        )
        
        memory = MemoryV2(cfg["memory"]["path"])
        
        input_dir = cfg["data"]["input_dir"]
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        with base_model.start_server(), critic_model.start_server():
            for input_file in input_files:
                logger.info(f"\n处理文件: {input_file}")
                
                input_path = os.path.join(input_dir, input_file)
                dataset = load_dataset(input_path)
                
                iteration = 1
                while iteration <= max_iterations: #外圈
                    logger.info(f"\n开始第 {iteration} 轮迭代...")
                    
                    results = single_iteration(
                        dataset=dataset,
                        base_model=base_model,
                        critic_model=critic_model,
                        memory=memory,
                        iteration=iteration,
                        batch_size=cfg["data"]["batch_size"]
                    )
                    
                    # 保存每轮结果
                    iter_output_path = os.path.join(
                        output_dir, 
                        f"iteration_{iteration}_{input_file}"
                    )
                    mmengine.dump(results, iter_output_path, indent=2)
                    
                    # 检查是否满足终止条件
                    if check_termination(dataset, memory, iteration):
                        logger.info(f"已达到质量要求，在第 {iteration} 轮终止迭代")
                        break
                    
                    if iteration == max_iterations:
                        logger.warning(f"达到最大迭代次数 {max_iterations}，强制终止")
                    
                    iteration += 1
                
                final_tuning_data = create_sft_data(dataset, memory, iteration)
                tuning_output_path = os.path.join(
                    output_dir,
                    f"tuning_data_{input_file}"
                )
                mmengine.dump(final_tuning_data, tuning_output_path, indent=2)
                
                logger.info(f"处理完成: {input_file}")
                logger.info(f"- 总迭代次数: {iteration}")
                logger.info(f"- 最终数据已保存到: {tuning_output_path}")
                
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise
    finally:
        memory.save_to_disk()

def create_sft_data(dataset: List[Dict], memory: Memory, final_iteration: int) -> List[Dict]:
    """
    创建最终的微调数据集
    
    Args:
        dataset: 原始数据集
        memory
        final_iteration: 最终收敛的迭代轮次
    
    Returns:
        List[Dict]: 用于微调的数据集，格式为 [{"conversations": [{"from": "human"/"gpt", "value": str}]}]
    """
    logger.info(f"生成微调数据集（使用第 {final_iteration} 轮结果）")
    tuning_data = []
    
    for conv_idx, conv in enumerate(dataset):
        conv_id = str(conv_idx + 1)
        turn_id = 1  # 当前只处理第一轮对话
        
        # 获取最佳响应
        best_response = memory.get_best_response(conv_id, turn_id)
        if best_response and best_response['score'] > 0:  # 只使用正面评价的响应
            # 获取原始query
            query = None
            for turn in conv["conversations"]:
                if turn["from"] == "human":
                    query = turn["value"]
                    break
            
            if query:
                tuning_data.append({
                    "conversations": [
                        {"from": "human", "value": query},
                        {"from": "gpt", "value": best_response['response']}
                    ],
                    "metadata": {
                        "thought": best_response['thought'],
                        "score": best_response['score']
                    }
                })
    
    logger.info(f"生成的微调数据集大小: {len(tuning_data)}")
    return tuning_data

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'run.log')),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    main() 