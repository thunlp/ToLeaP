import logging
import os
import mmengine
from models.base_model import BaseModel
from models.critic_model import CriticModel
from utils.memory_v2 import MemoryV2
from utils.data_loader import load_dataset
from run import single_iteration, check_termination, create_sft_data

# 设置基础日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multiple_iterations(max_iterations=3):
    # 创建输出目录
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    base_model = BaseModel(
        model="/bjzhyai03/workhome/niuboye/model/llama3_instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        use_api_model=False,
        port=8008
    )
    
    critic_model = CriticModel(
        model="/bjzhyai03/workhome/niuboye/model/llama3_instruct",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        use_api_model=False,
        port=8088
    )
    
    # 初始化内存
    memory = MemoryV2("xlammemo.json")
    
    # 加载测试数据
    dataset = load_dataset("/bjzhyai03/workhome/niuboye/pipline/Salesforce-xlam-function-calling-60k.json")
    
    try:
        iteration = 1
        while iteration <= max_iterations:
            logger.info(f"\n开始第 {iteration} 轮迭代...")
            
            # 运行单次迭代
            results = single_iteration(
                dataset=dataset,
                base_model=base_model,
                critic_model=critic_model,
                memory=memory,
                iteration=iteration,
                batch_size=2
            )
            
            # 保存每轮结果
            output_path = os.path.join(output_dir, f"iteration_{iteration}_results_toolace.json")
            mmengine.dump(results, output_path, indent=2)
            logger.info(f"第 {iteration} 轮结果已保存到: {output_path}")
            
            # 分析本轮结果
            # analyze_iteration_results(results, iteration)
            
            iteration += 1
        
            
    except Exception as e:
        logger.error(f"测试执行出错: {e}")
        raise
    finally:
        memory.save_to_disk()

def analyze_iteration_results(results, iteration):
    """分析每轮迭代的结果"""
    total_responses = len(results)
    positive_count = 0
    turn_stats = {}  # 按轮次统计
    retry_stats = {}  # 重试统计
    
    for result in results:
        turn_id = result.get('turn_id', 1)
        if turn_id not in turn_stats:
            turn_stats[turn_id] = {"total": 0, "positive": 0}
            retry_stats[turn_id] = {"total_retries": 0, "samples": 0}
            
        for turn in result['conversations']:
            if turn['from'] == 'gpt':
                metadata = turn['metadata']
                score = metadata['score']
                retry_count = metadata.get('retry_count', 0)
                
                turn_stats[turn_id]["total"] += 1
                if score == "Positive":
                    turn_stats[turn_id]["positive"] += 1
                    positive_count += 1
                
                retry_stats[turn_id]["total_retries"] += retry_count
                retry_stats[turn_id]["samples"] += 1
    
    logger.info(f"\n第 {iteration} 轮分析结果:")
    logger.info(f"- 总回复数: {total_responses}")
    logger.info(f"- 总体正面比例: {positive_count/total_responses*100:.1f}%")
    logger.info("- 各轮次统计:")
    for turn_id, stats in turn_stats.items():
        positive_ratio = stats["positive"] / stats["total"] * 100
        avg_retries = retry_stats[turn_id]["total_retries"] / retry_stats[turn_id]["samples"]
        logger.info(f"  轮次 {turn_id}:")
        logger.info(f"    - 正面率: {stats['positive']}/{stats['total']} ({positive_ratio:.1f}%)")
        logger.info(f"    - 平均重试次数: {avg_retries:.2f}")
    
    # 添加更多分析
    logger.info("\n详细统计:")
    for turn_id, stats in turn_stats.items():
        logger.info(f"\n轮次 {turn_id}:")
        logger.info(f"- 总尝试次数: {stats['total']}")
        logger.info(f"- 成功次数: {stats['positive']}")
        logger.info(f"- 成功率: {stats['positive']/stats['total']*100:.1f}%")
        logger.info(f"- 平均重试次数: {retry_stats[turn_id]['total_retries']/retry_stats[turn_id]['samples']:.2f}")


if __name__ == "__main__":
    test_multiple_iterations(max_iterations=3)