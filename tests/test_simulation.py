import numpy as np
import pytest

from bandit.simulation import ExperimentConfig, ExperimentResult, run_replication

CONFIG = ExperimentConfig(
    name="test", horizon=60, change_points=(30,), window=20, replications=2, grid_size=25
)


@pytest.fixture(scope="module")
def result():
    seeds = np.random.SeedSequence(CONFIG.seed).spawn(CONFIG.replications)
    return ExperimentResult(CONFIG, [run_replication(CONFIG, seed) for seed in seeds])


def test_regret_is_non_negative_and_bounded(result):
    for replication in result.replications:
        assert replication.regret.shape == (CONFIG.horizon,)
        assert np.all(replication.regret >= 0)
        assert np.all(replication.regret <= 1)


def test_replications_are_reproducible():
    seed = np.random.SeedSequence(7)
    first, second = run_replication(CONFIG, seed), run_replication(CONFIG, seed)
    np.testing.assert_array_equal(first.regret, second.regret)


def test_snapshots_only_hold_observations_from_the_final_window(result):
    for replication in result.replications:
        assert sum(len(arm.contexts) for arm in replication.arms) == CONFIG.window


def test_summary_shapes(result):
    assert result.cumulative_regret.shape == (CONFIG.replications, CONFIG.horizon)
    assert result.true_mean_rewards.shape == (CONFIG.grid_size, CONFIG.n_arms)
    assert result.estimated_best_arm.shape == (CONFIG.grid_size,)
    assert 0 <= result.best_arm_accuracy <= 1
    assert np.all(np.diff(result.uniform_policy_cumulative_regret) > 0)
