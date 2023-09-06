from typing import Callable

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import MinMaxScaler


DOMAIN_SAMPLES = 200  # precision of domain


class Reward:

    def __init__(
        self,
        index: int,
        shift: float,
        true_function: Callable[[np.ndarray, int], np.ndarray],
        bounds: tuple[float, float],
        reward_variance: float = 0.1,
        window_length: int = 0
    ) -> None:
        self.index = index
        self.shift = shift
        self.true_function = true_function
        self.bounds = bounds
        self.reward_variance = reward_variance
        self.window_length = window_length
        self.domain = np.linspace(
            start=self.bounds[0],
            stop=self.bounds[1],
            num=DOMAIN_SAMPLES
        ).reshape(-1, 1)
        self.scaler = MinMaxScaler()
        self.true_values = self.scaler.fit_transform(true_function(self.domain, shift))
        self.true_maximiser = self.domain[np.argmax(self.true_values)]
        self.true_maximum = self.scaler.transform(
            true_function(self.true_maximiser, shift).reshape(1, -1)
        )
        self.posterior = GaussianProcessRegressor(
            kernel=Matern(),
            alpha=0.5,
            n_restarts_optimizer=1
        )
        self.contexts = np.empty(0)
        self.rewards = np.empty(0)
        self.trainable_contexts = np.empty(0)
        self.trainable_rewards = np.empty(0)

    def _fit(self, context: np.ndarray, rewards: np.ndarray) -> None:
        ''' Retrain GP regression model '''
        self.posterior.fit(context, rewards)

    def play(self, context: np.ndarray) -> np.ndarray:
        ''' Play associated arm with given context, return observed reward and update GP '''
        self.contexts = np.append(self.contexts, context)
        reward = self.scaler.transform(
            self.true_function(context, self.shift)
        )+stats.norm.rvs(scale=self.reward_variance)
        self.rewards = np.append(self.rewards, reward)
        self.trainable_contexts = self.contexts
        self.trainable_rewards = self.rewards
        if self.window_length > 0:
            self.trainable_contexts = self.contexts[-self.window_length:]
            self.trainable_rewards = self.rewards[-self.window_length:]
        self._fit(
            self.trainable_contexts.reshape(-1, 1),
            self.trainable_rewards.reshape(-1, 1)
        )
        return reward

    def predict(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ''' Use current posterior to estimate reward mean and standard deviation (uncertainty) '''
        mean_prediction, std_prediction = self.posterior.predict(
            context, return_std=True
        )
        return mean_prediction, std_prediction

    def sample(self, context: np.ndarray) -> np.ndarray:
        ''' Sample rewards from the posterior given context '''
        sampled_reward = self.posterior.sample_y(context)
        return sampled_reward

    def reset(self, change_point: bool = False) -> None:
        ''' Clear context and reward history, re-initialise GP model '''
        if change_point:
            self.true_values = self.scaler.fit_transform(
                self.true_function(self.domain, self.shift)
            )
            self.true_maximiser = self.domain[np.argmax(self.true_values)]
            self.true_maximum = self.scaler.transform(
                self.true_function(self.true_maximiser, self.shift).reshape(1, -1)
            )
        else:
            self.posterior = GaussianProcessRegressor()
            self.contexts = np.empty(0)
            self.rewards = np.empty(0)
            self.trainable_contexts = np.empty(0)
            self.trainable_rewards = np.empty(0)

    def plot(self, confidence: float = 0.95, window_length: int = 0, file_name: str = 'reward.pdf') -> None:
        ''' Plot true reward function, observed reward samples, estimated posterior '''
        predicted_reward_mean, predicted_reward_std = self.predict(self.domain)
        plt.figure()
        plt.plot(
            self.domain,
            self.true_values,
            label='True reward function',
            linestyle='dotted',
            c='tab:blue'
        )
        if window_length > 0:
            plt.scatter(
                self.trainable_contexts[-window_length:],
                self.trainable_rewards[-window_length:],
                label=f'Observed rewards (last {window_length} observations)',
                c='tab:blue'
            )
        else:
            plt.scatter(
                self.trainable_contexts,
                self.trainable_rewards,
                label='Observed rewards',
                c='tab:blue'
            )
        plt.plot(
            self.domain,
            predicted_reward_mean,
            label='Predicted reward function',
            c='tab:orange'
        )
        plt.fill_between(
            self.domain.ravel(),
            predicted_reward_mean-stats.norm.ppf((1+confidence)/2)*predicted_reward_std,
            predicted_reward_mean+stats.norm.ppf((1+confidence)/2)*predicted_reward_std,
            alpha=0.5,
            label=f'{int(confidence*100)}% confidence interval',
            color='tab:orange'
        )
        plt.xlabel('Context')
        plt.ylabel('Reward')
        plt.ylim([np.min(self.true_values-0.5), np.max(self.true_values+0.5)])
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_name)
