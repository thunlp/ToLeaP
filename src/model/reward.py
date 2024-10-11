from abc import ABC


class RewardModel(ABC):
    def __init__(self):
        pass

    def cal_reward(self, state):
        pass


class PRM(RewardModel):
    def __init__(self):
        super().__init__()


class ORM(RewardModel):
    def __init__(self):
        super().__init__()


class KagglePRM(PRM):
    def __init__(self):
        super().__init__()


class KaggleORM(ORM):
    def __init__(self):
        super().__init__()