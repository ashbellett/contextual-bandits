# Standard libraries
from pathlib import Path
from time import time

# PyPI packages
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Local package
from bandit.examples import reward_function
from bandit.experiment import (
    create_folders,
    generate_arms,
    generate_domain,
    run
)

# Global settings
np.random.seed(1)
plt.rcParams.update({'figure.figsize': (8, 6)})
plt.rcParams.update({'font.size': 16})

# Global variables
DATA_PATH = Path('./figures')

# Static environment

# Simulation parameters
arm_count = 3  # arm space, 3 actions
dimension = 1  # 1-dimensional context values
bounds = (0, 1)  # context space bounded between 0 and 1
reward_variance = 0.1  # variance of reward random variables
window_length = 0  # number of observations to train GP model
horizon = 1000  # number of time steps to run simulation
replications = 30  # number of repeated experiment runs
change_points = []  # locations of change-point events

# Run simulation
create_folders(DATA_PATH)
domain = generate_domain(bounds)
arms_initial = generate_arms(
    true_reward_function=reward_function,
    arm_count=arm_count,
    dimension=dimension,
    bounds=bounds,
    reward_variance=reward_variance,
    window_length=window_length
)
start_time = time()
arms, cumulative_regret, estimated_best_actions, true_best_actions = run(
    arms=arms_initial,
    domain=domain,
    dimension=dimension,
    bounds=bounds,
    horizon=horizon,
    replications=replications,
    change_points=change_points
)
end_time = time()
print(f"Simulation took {round(end_time-start_time, 3)} s.")

# Regret
plt.figure()
plt.plot(cumulative_regret, label='Cumulative regret, Thompson sampling')
for index, change_point in enumerate(change_points):
    if index == 0:
        plt.plot(change_point, cumulative_regret[change_point], 'o', c='k', label='Change-point in reward function')
    else:
        plt.plot(change_point, cumulative_regret[change_point], 'o', c='k')
plt.vlines(
    x=change_points,
    ymin=[0]*len(change_points),
    ymax=[cumulative_regret[change_point] for change_point in change_points],
    colors='k',
    linestyles='dashed'
)
plt.xlabel('Time step')
plt.ylabel('Cumulative regret')
plt.legend()
plt.tight_layout()
plt.savefig(DATA_PATH / 'regret' / 'static.pdf')

# Optimal arm identification
plt.figure()
plt.plot(domain, estimated_best_actions, label='Estimated best action')
plt.plot(domain, true_best_actions, label='True best action')
plt.xlabel('Context')
plt.ylabel('Action')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(DATA_PATH / 'actions' / 'static.pdf')

# Reward estimation
for arm in arms:
    arm.reward.plot(file_name=DATA_PATH / 'reward' / f'static_{arm.index}.pdf')

# Dynamic environment (multiple changepoints)

# Simulation parameters
arm_count = 3  # arm space, 3 actions
dimension = 1  # 1-dimensional context values
bounds = (0, 1)  # context space bounded between 0 and 1
reward_variance = 0.1  # variance of reward random variables
window_length = 200  # number of observations to train GP model
horizon = 3000  # number of time steps to run simulation
replications = 30  # number of repeated experiment runs
change_points = [999, 1999]  # locations of change-point events

# Run simulation
create_folders(DATA_PATH)
domain = generate_domain(bounds)
arms_initial = generate_arms(
    true_reward_function=reward_function,
    arm_count=arm_count,
    dimension=dimension,
    bounds=bounds,
    reward_variance=reward_variance,
    window_length=window_length
)
start_time = time()
arms, cumulative_regret, estimated_best_actions, true_best_actions = run(
    arms=arms_initial,
    domain=domain,
    dimension=dimension,
    bounds=bounds,
    horizon=horizon,
    replications=replications,
    change_points=change_points
)
end_time = time()
print(f"Simulation took {round(end_time-start_time, 3)} s.")

# Regret
plt.figure()
plt.plot(cumulative_regret, label='Cumulative regret, Thompson sampling')
for index, change_point in enumerate(change_points):
    if index == 0:
        plt.plot(change_point, cumulative_regret[change_point], 'o', c='k', label='Change-point in reward function')
    else:
        plt.plot(change_point, cumulative_regret[change_point], 'o', c='k')
plt.vlines(
    x=change_points,
    ymin=[0]*len(change_points),
    ymax=[cumulative_regret[change_point] for change_point in change_points],
    colors='k',
    linestyles='dashed'
)
plt.xlabel('Time step')
plt.ylabel('Cumulative regret')
plt.legend()
plt.tight_layout()
plt.savefig(DATA_PATH / 'regret' / 'dynamic.pdf')

# Optimal arm identification
plt.figure()
plt.plot(domain, estimated_best_actions, label='Estimated best action')
plt.plot(domain, true_best_actions, label='True best action')
plt.xlabel('Context')
plt.ylabel('Action')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(DATA_PATH / 'actions' / 'dynamic.pdf')

# Reward estimation
for arm in arms:
    arm.reward.plot(file_name=DATA_PATH / 'reward' / f'dynamic_{arm.index}.pdf')
