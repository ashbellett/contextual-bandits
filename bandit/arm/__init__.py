import numpy as np

from bandit.reward import Reward


class Arm:

    def __init__(self, index: int, reward: Reward, context_dimension: int) -> None:
        self.index = index
        self.reward = reward
        self.context_dimension = context_dimension
        self.play_count = 0
        self.context_history = np.empty((0, context_dimension))
        self.reward_history = np.empty(0)

    def play(self, context: np.ndarray) -> float:
        ''' Play arm given context, update arm statistics and return observed reward '''
        reward = self.reward.play(context)
        self.play_count += 1
        self.context_history = np.vstack((self.context_history, context))
        self.reward_history = np.append(self.reward_history, reward)
        return reward

    def reset(self) -> None:
        ''' Clear context and reward history and reset underlying reward state '''
        self.play_count = 0
        self.context_history = np.empty((0, self.context_dimension))
        self.reward_history = np.empty(0)
        self.reward.reset()
