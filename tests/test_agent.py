import numpy as np

from bandit.agent import GPThompsonSampling


def test_sliding_window_keeps_only_recent_rounds():
    agent = GPThompsonSampling(n_arms=1, rng=np.random.default_rng(0), window=5)
    for t in range(10):
        agent.update(0, context=t / 10, reward=1.0, t=t)
    window = agent.window_slice(0, t=10)
    assert agent.histories[0].times[window] == [5, 6, 7, 8, 9]


def test_arm_reverts_to_prior_once_its_window_is_empty():
    agent = GPThompsonSampling(n_arms=1, rng=np.random.default_rng(0), window=5)
    for t in range(3):
        agent.update(0, context=0.5, reward=1.0, t=t)
    recent_mean, recent_std = agent.posterior(0, np.array([0.5]), t=3)
    stale_mean, stale_std = agent.posterior(0, np.array([0.5]), t=100)
    assert recent_mean[0] > 0.8
    assert stale_mean[0] == 0.5
    assert stale_std[0] > recent_std[0]


def test_thompson_samples_vary_between_calls():
    agent = GPThompsonSampling(n_arms=1, rng=np.random.default_rng(0))
    samples = {agent.sample_mean_reward(0, 0.5, t=0) for _ in range(5)}
    assert len(samples) == 5


def test_selects_arm_that_is_clearly_better():
    agent = GPThompsonSampling(n_arms=2, rng=np.random.default_rng(0))
    for t, context in enumerate(np.linspace(0, 1, 30)):
        agent.update(0, context, 0.1, 2 * t)
        agent.update(1, context, 0.9, 2 * t + 1)
    choices = [agent.select_arm(context, t=60) for context in np.linspace(0, 1, 20)]
    assert choices == [1] * 20
