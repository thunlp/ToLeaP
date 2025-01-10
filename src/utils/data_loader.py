import os
import json
from typing import Dict, List
import mmengine

def is_valid_conversation(conversation: Dict) -> bool:
    """检查对话是否符合human-gpt交替格式"""
    turns = conversation.get("conversations", [])
    if not turns:
        return False
        
    if turns[0]["from"] != "human":
        return False
        
    for i in range(len(turns)):
        expected_role = "human" if i % 2 == 0 else "gpt"
        if turns[i]["from"] != expected_role:
            return False
            
    return True

def load_dataset(file_path: str) -> List[Dict]:
    """加载数据集并过滤无效对话"""
    try:
        data = mmengine.load(file_path)
        if not isinstance(data, list):
            data = [data]
            
        # 过滤无效对话
        valid_data = []
        invalid_count = 0
        empty_count = 0
        
        for conv in data:
            # 检查是否为空对话
            if not conv.get("conversations"):
                empty_count += 1
                continue
                
            # 检查是否符合格式要求
            if not is_valid_conversation(conv):
                invalid_count += 1
                continue
                
            valid_data.append(conv)
        
        print(f"数据集统计 {file_path}:")
        print(f"- 总对话数: {len(data)}")
        print(f"- 空对话数: {empty_count}")
        print(f"- 格式无效数: {invalid_count}")
        print(f"- 有效对话数: {len(valid_data)}")
        
        return valid_data
        
    except Exception as e:
        print(f"加载数据集错误 {file_path}: {e}")
        return [] 