"""Piecewise-stationary contextual bandit with smooth, normalised reward curves."""

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

RewardCurve = Callable[[np.ndarray, float], np.ndarray]


def oscillating_reward(contexts: np.ndarray, phase: float) -> np.ndarray:
    """u·sin(u) with u = 10x + φπ/2: several local maxima whose height grows with u."""
    u = 10 * contexts + phase * np.pi / 2
    return u * np.sin(u)


@dataclass(frozen=True)
class PiecewiseStationaryEnvironment:
    """Arms whose mean-reward curves are fixed between change points.

    In regime r (the number of change points at or before t), arm a has mean
    reward ``reward_curve(x, a + r * n_arms)`` rescaled to [0, 1] over the
    context interval. Observed rewards add Gaussian noise with ``noise_std``.
    """

    n_arms: int
    change_points: tuple[int, ...] = ()
    noise_std: float = 0.1
    context_bounds: tuple[float, float] = (0.0, 1.0)
    reward_curve: RewardCurve = oscillating_reward
    normalisation_grid_size: int = 2001

    def __post_init__(self) -> None:
        if list(self.change_points) != sorted(set(self.change_points)):
            raise ValueError("change_points must be strictly increasing")

    def regime(self, t: int) -> int:
        return bisect_right(self.change_points, t)

    def mean_rewards(self, contexts: np.ndarray, t: int) -> np.ndarray:
        """Expected reward of every arm at each context, shape (len(contexts), n_arms)."""
        contexts = np.atleast_1d(contexts)
        grid = np.linspace(*self.context_bounds, self.normalisation_grid_size)
        regime = self.regime(t)
        columns = []
        for arm in range(self.n_arms):
            phase = arm + regime * self.n_arms
            reference = self.reward_curve(grid, phase)
            low, high = reference.min(), reference.max()
            columns.append((self.reward_curve(contexts, phase) - low) / (high - low))
        return np.column_stack(columns)

    def sample_context(self, rng: np.random.Generator) -> float:
        return rng.uniform(*self.context_bounds)

    def pull(self, arm: int, context: float, t: int, rng: np.random.Generator) -> float:
        mean = self.mean_rewards(context, t)[0, arm]
        return mean + rng.normal(scale=self.noise_std)
