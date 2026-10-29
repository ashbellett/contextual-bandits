"""Replicated simulations of GP Thompson sampling and their summary statistics."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
from threadpoolctl import threadpool_limits

from bandit.agent import GPThompsonSampling
from bandit.environment import PiecewiseStationaryEnvironment


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    n_arms: int = 3
    horizon: int = 1000
    change_points: tuple[int, ...] = ()
    noise_std: float = 0.1
    window: int | None = None
    optimise_every: int = 25
    replications: int = 30
    seed: int = 1
    grid_size: int = 200

    @property
    def environment(self) -> PiecewiseStationaryEnvironment:
        return PiecewiseStationaryEnvironment(
            n_arms=self.n_arms, change_points=self.change_points, noise_std=self.noise_std
        )

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(*self.environment.context_bounds, self.grid_size)


@dataclass(frozen=True)
class ArmSnapshot:
    """One arm's posterior and the observations it was conditioned on, after the last round."""

    contexts: np.ndarray
    rewards: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray


@dataclass(frozen=True)
class ReplicationResult:
    regret: np.ndarray
    arms: list[ArmSnapshot]


def run_replication(config: ExperimentConfig, seed: np.random.SeedSequence) -> ReplicationResult:
    """Play one episode; regret at round t is max_a μ_a(x_t) - μ_{a_t}(x_t)."""
    rng = np.random.default_rng(seed)
    environment = config.environment
    agent = GPThompsonSampling(config.n_arms, rng, window=config.window, optimise_every=config.optimise_every)
    regret = np.empty(config.horizon)
    for t in range(config.horizon):
        context = environment.sample_context(rng)
        arm = agent.select_arm(context, t)
        reward = environment.pull(arm, context, t, rng)
        agent.update(arm, context, reward, t)
        means = environment.mean_rewards(context, t)[0]
        regret[t] = means.max() - means[arm]

    snapshots = []
    for arm in range(config.n_arms):
        mean, std = agent.posterior(arm, config.grid, config.horizon)
        window = agent.window_slice(arm, config.horizon)
        history = agent.histories[arm]
        snapshots.append(
            ArmSnapshot(
                contexts=np.array(history.contexts[window]),
                rewards=np.array(history.rewards[window]),
                posterior_mean=mean,
                posterior_std=std,
            )
        )
    return ReplicationResult(regret, snapshots)


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    replications: list[ReplicationResult]

    @cached_property
    def cumulative_regret(self) -> np.ndarray:
        """Shape (replications, horizon)."""
        return np.cumsum([r.regret for r in self.replications], axis=1)

    @cached_property
    def true_mean_rewards(self) -> np.ndarray:
        """Mean rewards in the final regime, shape (grid_size, n_arms)."""
        return self.config.environment.mean_rewards(self.config.grid, self.config.horizon - 1)

    @cached_property
    def average_posterior_mean(self) -> np.ndarray:
        """Final posterior means averaged over replications, shape (grid_size, n_arms)."""
        return np.mean([[arm.posterior_mean for arm in r.arms] for r in self.replications], axis=0).T

    @property
    def true_best_arm(self) -> np.ndarray:
        return np.argmax(self.true_mean_rewards, axis=1)

    @property
    def estimated_best_arm(self) -> np.ndarray:
        return np.argmax(self.average_posterior_mean, axis=1)

    @property
    def best_arm_accuracy(self) -> float:
        """Share of (replication, context) pairs where the posterior-mean argmax is optimal."""
        estimates = np.array(
            [np.argmax([arm.posterior_mean for arm in r.arms], axis=0) for r in self.replications]
        )
        return float(np.mean(estimates == self.true_best_arm))

    @cached_property
    def uniform_policy_cumulative_regret(self) -> np.ndarray:
        """Expected cumulative regret of choosing arms uniformly at random."""
        environment, grid = self.config.environment, self.config.grid
        per_regime = {}
        for t in [0, *self.config.change_points]:
            means = environment.mean_rewards(grid, t)
            per_regime[environment.regime(t)] = np.mean(means.max(axis=1) - means.mean(axis=1))
        regret = [per_regime[environment.regime(t)] for t in range(self.config.horizon)]
        return np.cumsum(regret)


def _limit_to_one_blas_thread() -> None:
    threadpool_limits(limits=1)


def run_experiment(config: ExperimentConfig, workers: int | None = None) -> ExperimentResult:
    """Run independent, reproducibly seeded replications in parallel processes.

    Each process is limited to one BLAS thread so that parallel replications do not
    oversubscribe the CPU.
    """
    seeds = np.random.SeedSequence(config.seed).spawn(config.replications)
    with ProcessPoolExecutor(max_workers=workers, initializer=_limit_to_one_blas_thread) as executor:
        replications = list(executor.map(partial(run_replication, config), seeds))
    return ExperimentResult(config, replications)
