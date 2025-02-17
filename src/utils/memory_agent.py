import json
import os
from typing import List

class MultiAgentMemory:
    def __init__(self, save_path: str = "multiagent_memory.json"):
        """
        Initialize memory system for multi-agent conversations
        
        Updated storage format:
        {
            "conversations": [
                {
                    "system_instruction": str,
                    "turns": [
                        {
                            "query": str,
                            "label": str,
                            "interactions": [
                                {
                                    "type": "base_model" | "agent",
                                    "thought": str,  # For base_model type
                                    "agent_to_call": str,  # For base_model type
                                    "agent": str,  # For agent type
                                    "instruction": str,  # For agent type
                                    "response": str  # For agent type
                                }
                            ],
                            "final_answer": str
                        }
                    ]
                }
            ]
        }
        """
        self.memory = []
        self.save_path = save_path
        self._load_memory()
        
    def _load_memory(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                print(f"Loaded {len(self.memory)} conversations from {self.save_path}")
            except Exception as e:
                print(f"Failed to load memory file: {e}")
                self.memory = []

    def save_to_disk(self):
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=4)
            # print(f"Saved {len(self.memory['conversations'])} conversations to {self.save_path}")
        except Exception as e:
            print(f"Failed to save memory file: {e}")

    def start_conversation(self, system_instruction: str):
        """Start tracking a new conversation"""
        new_conversation = {
            "system_instruction": system_instruction,
            "turns": []
        }
        self.memory.append(new_conversation)
        self.save_to_disk()
        return len(self.memory) - 1  # Return conversation index

    def add_turn(self, conv_idx: int, query: str, label: str):
        """Add a new turn to a conversation"""
        self.memory[conv_idx]["turns"].append({
            "query": query,
            "label": label,
            "interactions": [],
            "final_answer": None
        })
        self.save_to_disk()
        return len(self.memory[conv_idx]["turns"]) - 1  # Return turn index

    def add_base_model_thought(self, conv_idx: int, turn_idx: int, thought: str, agent_to_call: str):
        """Record a base model's thought and agent selection"""
        self.memory[conv_idx]["turns"][turn_idx]["interactions"].append({
            "type": "base_model",
            "thought": thought,
            "agent_to_call": agent_to_call
        })
        self.save_to_disk()

    def add_agent_interaction(self, conv_idx: int, turn_idx: int, agent: str, instruction: str, response: str):
        """Record an agent interaction within a turn"""
        self.memory[conv_idx]["turns"][turn_idx]["interactions"].append({
            "type": "agent",
            "agent": agent,
            "instruction": instruction,
            "response": response
        })
        self.save_to_disk()

    def set_final_answer(self, conv_idx: int, turn_idx: int, answer: str):
        """Set the final answer for a turn"""
        self.memory[conv_idx]["turns"][turn_idx]["final_answer"] = answer
        self.save_to_disk()

    def add_base_model_error_response(self, conv_idx: int, turn_idx: int, raw_response: str):
        """Record a base model's raw response when parsing fails"""
        self.memory[conv_idx]["turns"][turn_idx]["interactions"].append({
            "type": "base_model_error",
            "raw_response": raw_response
        })
        self.save_to_disk()
