"""Gaussian-process posterior over one arm's mean-reward curve."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, WhiteKernel

JITTER = 1e-8


def default_kernel() -> Kernel:
    """Signal variance × Matérn-5/2 + observation noise; all three are learned by ML-II."""
    signal = ConstantKernel(0.1, constant_value_bounds=(1e-3, 10.0)) * Matern(
        length_scale=0.2, length_scale_bounds=(1e-2, 1.0), nu=2.5
    )
    noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
    return signal + noise


class GaussianProcessModel:
    """GP regression with a constant prior mean and a predictive for the latent curve f.

    ``predict`` returns the posterior of f(x), not of a noisy observation y = f(x) + ε,
    because Thompson sampling must sample the mean reward.
    """

    def __init__(self, prior_mean: float = 0.5, kernel: Kernel | None = None) -> None:
        self.prior_mean = prior_mean
        self.kernel = kernel if kernel is not None else default_kernel()
        self._regressor: GaussianProcessRegressor | None = None

    @property
    def signal_kernel(self) -> Kernel:
        return self.kernel.k1

    @property
    def noise_variance(self) -> float:
        return self.kernel.k2.noise_level

    def condition_on(self, contexts: np.ndarray, rewards: np.ndarray, optimise_hyperparameters: bool) -> None:
        if len(contexts) == 0:
            self._regressor = None
            return
        regressor = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=JITTER,
            optimizer="fmin_l_bfgs_b" if optimise_hyperparameters else None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            regressor.fit(np.reshape(contexts, (-1, 1)), np.asarray(rewards) - self.prior_mean)
        self.kernel = regressor.kernel_
        self._regressor = regressor

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of f at each context."""
        inputs = np.reshape(contexts, (-1, 1))
        if self._regressor is None:
            mean = np.zeros(len(inputs))
            variance = self.signal_kernel.diag(inputs)
        else:
            mean, observation_std = self._regressor.predict(inputs, return_std=True)
            variance = observation_std**2 - self.noise_variance
        return mean + self.prior_mean, np.sqrt(np.clip(variance, 0.0, None))
