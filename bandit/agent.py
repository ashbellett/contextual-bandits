"""Thompson sampling with one Gaussian-process reward model per arm."""

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np

from bandit.gaussian_process import GaussianProcessModel


@dataclass
class ArmHistory:
    times: list[int] = field(default_factory=list)
    contexts: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    pulls_at_last_optimisation: int | None = None

    @property
    def pulls(self) -> int:
        return len(self.times)


class GPThompsonSampling:
    """Contextual GP Thompson sampling, optionally on a sliding window of recent rounds.

    With ``window=None`` every past observation is used (stationary model). With
    ``window=w`` the posterior at round t only uses observations from rounds
    t - w, ..., t - 1, so an arm that has not been pulled recently reverts to its
    prior and gets re-explored after a change in the environment.

    Kernel hyperparameters are re-learned by maximising the marginal likelihood
    every ``optimise_every`` pulls of an arm, once it has ``min_observations_to_optimise``
    observations in its window; in between, the posterior is updated with fixed
    hyperparameters.
    """

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        window: int | None = None,
        optimise_every: int = 25,
        min_observations_to_optimise: int = 10,
        prior_mean: float = 0.5,
    ) -> None:
        self.n_arms = n_arms
        self.rng = rng
        self.window = window
        self.optimise_every = optimise_every
        self.min_observations_to_optimise = min_observations_to_optimise
        self.models = [GaussianProcessModel(prior_mean) for _ in range(n_arms)]
        self.histories = [ArmHistory() for _ in range(n_arms)]
        self._conditioned_on: list[tuple[int, int]] = [(0, 0)] * n_arms

    def select_arm(self, context: float, t: int) -> int:
        samples = [self.sample_mean_reward(arm, context, t) for arm in range(self.n_arms)]
        return int(np.argmax(samples))

    def sample_mean_reward(self, arm: int, context: float, t: int) -> float:
        mean, std = self.posterior(arm, np.array([context]), t)
        return self.rng.normal(mean[0], std[0])

    def update(self, arm: int, context: float, reward: float, t: int) -> None:
        history = self.histories[arm]
        history.times.append(t)
        history.contexts.append(context)
        history.rewards.append(reward)

    def posterior(self, arm: int, contexts: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and std of arm's mean reward, given what is known before round t."""
        self._condition_on_window(arm, t)
        return self.models[arm].predict(contexts)

    def window_slice(self, arm: int, t: int) -> slice:
        history = self.histories[arm]
        end = bisect_left(history.times, t)
        start = 0 if self.window is None else bisect_left(history.times, t - self.window)
        return slice(start, end)

    def _condition_on_window(self, arm: int, t: int) -> None:
        window = self.window_slice(arm, t)
        if (window.start, window.stop) == self._conditioned_on[arm]:
            return
        history = self.histories[arm]
        last = history.pulls_at_last_optimisation
        optimise = window.stop - window.start >= self.min_observations_to_optimise and (
            last is None or history.pulls - last >= self.optimise_every
        )
        self.models[arm].condition_on(
            np.array(history.contexts[window]),
            np.array(history.rewards[window]),
            optimise_hyperparameters=optimise,
        )
        if optimise:
            history.pulls_at_last_optimisation = history.pulls
        self._conditioned_on[arm] = (window.start, window.stop)
