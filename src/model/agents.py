from abc import ABC
from typing import List


class Agent(ABC):
    def __init__(self,
                 profile:str = None,
                 action_space:List = [], # output language, pictures, videos, audio, function calls, ...
                 ):
        self.profile = profile
        self.action_space = action_space


class CriticAgent(Agent):
    def __init__(self, 
                 profile = None, 
                 action_space = [],
                 ):
        super().__init__(profile, action_space)

    def cal_reward_value(self, language_input):
        pass