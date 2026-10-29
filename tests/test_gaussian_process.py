import numpy as np
import pytest

from bandit.gaussian_process import GaussianProcessModel


def test_prior_is_constant_mean_with_signal_variance():
    model = GaussianProcessModel(prior_mean=0.5)
    mean, std = model.predict(np.linspace(0, 1, 4))
    np.testing.assert_allclose(mean, 0.5)
    np.testing.assert_allclose(std, np.sqrt(model.signal_kernel.diag(np.zeros((1, 1)))[0]))


def test_predictive_excludes_observation_noise():
    rng = np.random.default_rng(0)
    contexts = rng.uniform(0, 1, 200)
    rewards = np.sin(6 * contexts) / 2 + 0.5 + rng.normal(0, 0.1, 200)
    model = GaussianProcessModel()
    model.condition_on(contexts, rewards, optimise_hyperparameters=True)

    grid = np.linspace(0.05, 0.95, 20)
    mean, std = model.predict(grid)
    np.testing.assert_allclose(mean, np.sin(6 * grid) / 2 + 0.5, atol=0.06)
    assert model.noise_variance == pytest.approx(0.01, rel=0.4)
    assert np.all(std < 0.05)


def test_conditioning_on_nothing_restores_prior():
    model = GaussianProcessModel()
    model.condition_on(np.array([0.5]), np.array([1.0]), optimise_hyperparameters=False)
    model.condition_on(np.array([]), np.array([]), optimise_hyperparameters=False)
    mean, _ = model.predict(np.array([0.5]))
    assert mean[0] == pytest.approx(0.5)
