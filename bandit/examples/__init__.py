import math

import numpy as np


def reward_function(observation: np.ndarray, shift: float = 0) -> np.ndarray:
    ''' Example of a reward function with multiple local maxima '''
    reward_value = (10*observation+shift*math.pi/2)*np.sin(10*observation+shift*math.pi/2)
    return reward_value
