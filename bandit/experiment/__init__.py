import copy
import os
from typing import Callable, Union

import numpy as np

from bandit.arm import Arm
from bandit.reward import Reward


DOMAIN_SAMPLES = 200  # precision of domain


def generate_domain(bounds: tuple[float, float]) -> np.ndarray:
    ''' Generate equally-spaced 1D vector to represent domain '''
    domain = np.linspace(
        start=bounds[0],
        stop=bounds[1],
        num=DOMAIN_SAMPLES
    ).reshape(-1, 1)
    return domain


def generate_arms(
    true_reward_function: Callable[[np.ndarray, float], np.ndarray],
    arm_count: int,
    dimension: int,
    bounds: tuple[int, int],
    reward_variance: float,
    window_length: int = 0
) -> list[Arm]:
    ''' Generate action (arm) space '''
    arms = []
    for index in range(arm_count):
        reward_function = Reward(
            index,
            shift=index,
            true_function=true_reward_function,
            bounds=bounds,
            reward_variance=reward_variance,
            window_length=window_length
        )
        arm = Arm(index, reward_function, dimension)
        arms.append(arm)
    return arms


def calculate_regret(
    arms: list[Arm],
    dimension: int,
    bounds: tuple[int, int],
    horizon: int,
    change_points: list[Union[int, None]]
) -> tuple[list[Arm], np.ndarray]:
    ''' Apply Thompson sampling to play arms and calculate regret '''
    regrets = np.zeros(horizon)
    reward_variance = arms[0].reward.reward_variance
    for time_index in range(horizon):
        if time_index in change_points:
            arms = change_rewards(
                arms,
                reward_variance=reward_variance,
                offset=change_points.index(time_index)
            )
        context = np.random.uniform(low=bounds[0], high=bounds[1], size=dimension)
        estimated_rewards = []
        for arm in arms:
            estimated_reward = arm.reward.sample(context.reshape(-1, 1))
            estimated_rewards.append(estimated_reward)
        estimated_best_arm = arms[np.argmax(estimated_rewards)]
        observed_reward = estimated_best_arm.play(context.reshape(-1, 1))
        true_best_reward = np.max([
            arm.reward.scaler.transform(
                arm.reward.true_function(context, arm.reward.shift).reshape(1, -1)
            ) for arm in arms
        ])
        regret = true_best_reward-observed_reward
        regrets[time_index] = regret.item()
    return arms, regrets


def calculate_rewards(
    arms: list[Arm],
    domain: np.ndarray
) -> tuple[list[Arm], np.ndarray, np.ndarray]:
    ''' Calculate estimated and true best rewards for all contexts '''
    estimated_rewards = np.zeros((len(arms), len(domain)))
    best_rewards = np.zeros((len(arms), len(domain)))
    for arm in arms:
        for index, value in enumerate(domain):
            estimated_reward, _ = arm.reward.predict(value.reshape(1, -1))
            best_reward = arm.reward.scaler.transform(
                arm.reward.true_function(
                    value.reshape(1, -1), arm.reward.shift
                ).reshape(1, -1)
            )
            estimated_rewards[arm.index, index] = estimated_reward.item()
            best_rewards[arm.index, index] = best_reward.item()
    return arms, estimated_rewards, best_rewards


def run_experiment(
    arms: list[Arm],
    domain: np.ndarray,
    dimension: int,
    bounds: tuple[int, int],
    horizon: int,
    change_points: list[Union[int, None]]
) -> tuple[list[Arm], np.ndarray, np.ndarray, np.ndarray]:
    ''' Play arms and calculate regret, estimated rewards and best rewards '''
    arms, regrets = calculate_regret(arms, dimension, bounds, horizon, change_points)
    arms, estimated_rewards, best_rewards = calculate_rewards(arms, domain)
    return arms, regrets, estimated_rewards, best_rewards


def average_experiment(
    arms: list[Arm],
    domain: np.ndarray,
    dimension: int,
    bounds: tuple[int, int],
    horizon: int,
    replications: int,
    change_points: list[Union[int, None]],
    reset_arms: bool
) -> tuple[list[Arm], np.ndarray, np.ndarray, np.ndarray]:
    ''' Replicate experiment and average calculated results '''
    regrets_average = np.zeros(horizon)
    expected_rewards_average = np.zeros((len(arms), len(domain)))
    if not reset_arms:
        frozen_arms = copy.deepcopy(arms)
    for _ in range(replications):
        if reset_arms:
            for arm in arms:
                arm.reset()
        else:
            arms = copy.deepcopy(frozen_arms)
        arms, regrets, estimated_rewards, best_rewards = run_experiment(
            arms, domain, dimension, bounds, horizon, change_points
        )
        regrets_average += regrets
        expected_rewards_average += estimated_rewards
    regrets_average /= replications
    expected_rewards_average /= replications
    return arms, regrets_average, expected_rewards_average, best_rewards


def change_rewards(arms: list[Arm], reward_variance: float, offset: int = 0) -> list[Arm]:
    ''' Update arms' reward functions '''
    new_arms = copy.deepcopy(arms)
    true_reward_function = new_arms[0].reward.true_function
    bounds = new_arms[0].reward.bounds
    window_length = new_arms[0].reward.window_length
    for index, arm in enumerate(new_arms):
        reward_function = Reward(
            index,
            shift=index+(1+offset)*len(new_arms),
            true_function=true_reward_function,
            bounds=bounds,
            reward_variance=reward_variance,
            window_length=window_length
        )
        reward_function.posterior = arm.reward.posterior
        reward_function.contexts = arm.reward.contexts
        reward_function.rewards = arm.reward.rewards
        reward_function.trainable_contexts = arm.reward.trainable_contexts
        reward_function.trainable_rewards = arm.reward.trainable_rewards
        new_arms[index].reward = reward_function
        new_arms[index].reward.reset(change_point=True)
    return new_arms


def run(
    arms: list[Arm],
    domain: np.ndarray,
    dimension: int,
    bounds: tuple[int, int],
    horizon: int = 1000,
    replications: int = 30,
    change_points: list[Union[int, None]] = [None],
    reset_arms: bool = True
) -> tuple[list[Arm], np.ndarray, np.ndarray, np.ndarray]:
    ''' Run end-to-end simulation '''
    arms_updated, regrets_average, expected_rewards_average, best_rewards = average_experiment(
        arms, domain, dimension, bounds, horizon, replications, change_points, reset_arms
    )
    cumulative_regret = np.cumsum(regrets_average)
    estimated_best_actions = np.argmax(expected_rewards_average, axis=0).astype(int)
    true_best_actions = np.argmax(best_rewards, axis=0).astype(int)
    return arms_updated, cumulative_regret, estimated_best_actions, true_best_actions


def create_folders(data_path: str) -> None:
    ''' Create folders to save figures '''
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    for folder in ['regret', 'actions', 'reward']:
        if not os.path.exists(data_path / folder):
            os.mkdir(data_path / folder)
