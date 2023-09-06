from typing import Callable

from matplotlib import pyplot as plt, cm
import numpy as np


DOMAIN_SAMPLES = 200  # precision of domain


def plot_3d(
    function: Callable[[np.ndarray], np.ndarray],
    index: int,
    bounds: tuple[float, float]
) -> None:
    ''' Plot a 3D surface '''
    x = np.arange(bounds[0], bounds[1], 1/DOMAIN_SAMPLES)
    y = np.arange(bounds[0], bounds[1], 1/DOMAIN_SAMPLES)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Z = function(R, index)
    figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
    surface = axes.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0
    )
    figure.colorbar(surface, shrink=0.5, aspect=5)
    plt.show()
