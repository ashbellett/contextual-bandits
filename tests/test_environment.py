import numpy as np
import pytest

from bandit.environment import PiecewiseStationaryEnvironment


def test_mean_rewards_are_normalised_to_unit_interval():
    environment = PiecewiseStationaryEnvironment(n_arms=3)
    grid = np.linspace(0, 1, environment.normalisation_grid_size)
    means = environment.mean_rewards(grid, t=0)
    assert means.shape == (len(grid), 3)
    np.testing.assert_allclose(means.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(means.max(axis=0), 1.0, atol=1e-12)


def test_regime_changes_at_change_points():
    environment = PiecewiseStationaryEnvironment(n_arms=2, change_points=(10, 20))
    assert [environment.regime(t) for t in (0, 9, 10, 19, 20, 99)] == [0, 0, 1, 1, 2, 2]
    contexts = np.linspace(0, 1, 5)
    assert not np.allclose(environment.mean_rewards(contexts, 9), environment.mean_rewards(contexts, 10))


def test_change_points_must_be_increasing():
    with pytest.raises(ValueError):
        PiecewiseStationaryEnvironment(n_arms=2, change_points=(20, 10))


def test_pull_adds_zero_mean_noise():
    environment = PiecewiseStationaryEnvironment(n_arms=2, noise_std=0.1)
    rng = np.random.default_rng(0)
    rewards = [environment.pull(1, 0.3, 0, rng) for _ in range(4000)]
    mean = environment.mean_rewards(0.3, 0)[0, 1]
    assert np.mean(rewards) == pytest.approx(mean, abs=0.01)
    assert np.std(rewards) == pytest.approx(0.1, rel=0.05)
