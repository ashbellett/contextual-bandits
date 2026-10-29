"""Matplotlib figures for an experiment result, built without pyplot's global state."""

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

from bandit.simulation import ExperimentResult

ARM_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": MUTED_INK,
    "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "figure.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "svg.fonttype": "none",
    "svg.hashsalt": "contextual-bandits",
}


def arm_colour(arm: int) -> str:
    return ARM_COLOURS[arm % len(ARM_COLOURS)]


def plot_cumulative_regret(result: ExperimentResult) -> Figure:
    """Mean cumulative regret with a 95% confidence band for the mean."""
    regret = result.cumulative_regret
    mean = regret.mean(axis=0)
    half_width = norm.ppf(0.975) * regret.std(axis=0, ddof=1) / np.sqrt(len(regret))
    rounds = np.arange(1, len(mean) + 1)

    figure = Figure(figsize=(8, 3.2), layout="constrained")
    axes = figure.subplots()
    axes.fill_between(
        rounds, mean - half_width, mean + half_width, color=ARM_COLOURS[0], alpha=0.2, linewidth=0
    )
    axes.plot(rounds, mean, color=ARM_COLOURS[0], label="GP Thompson sampling (mean)")
    for index, change_point in enumerate(result.config.change_points):
        axes.axvline(
            change_point,
            color=MUTED_INK,
            linestyle=":",
            linewidth=1,
            label="Change point" if index == 0 else None,
        )
    axes.set_xlim(0, len(mean))
    axes.set_ylim(bottom=0)
    axes.set_xlabel("Round")
    axes.set_ylabel("Cumulative regret")
    axes.legend(loc="upper left")
    return figure


def plot_best_arm(result: ExperimentResult) -> Figure:
    """Which arm is optimal at each context: truth versus the learned posterior means."""
    n_arms = result.config.n_arms
    grid = result.config.grid
    rows = np.vstack([result.true_best_arm, result.estimated_best_arm])
    colour_map = ListedColormap([arm_colour(arm) for arm in range(n_arms)])

    figure = Figure(figsize=(8, 1.9), layout="constrained")
    axes = figure.subplots()
    step = grid[1] - grid[0]
    axes.imshow(
        rows,
        aspect="auto",
        cmap=colour_map,
        vmin=-0.5,
        vmax=n_arms - 0.5,
        interpolation="nearest",
        extent=(grid[0] - step / 2, grid[-1] + step / 2, 1.5, -0.5),
    )
    axes.axhline(0.5, color=SURFACE, linewidth=3)
    axes.set_yticks([0, 1], ["True best arm", "Estimated best arm"])
    axes.set_xlabel("Context x")
    axes.grid(False)
    axes.tick_params(axis="y", length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    handles = [Patch(color=arm_colour(arm), label=f"Arm {arm}") for arm in range(n_arms)]
    figure.legend(handles=handles, loc="outside upper left", ncols=n_arms)
    return figure


def plot_arm_posteriors(result: ExperimentResult, replication: int = 0, credibility: float = 0.95) -> Figure:
    """Each arm's true mean reward, observations and posterior at the end of one replication."""
    n_arms = result.config.n_arms
    grid = result.config.grid
    z = norm.ppf((1 + credibility) / 2)
    snapshot = result.replications[replication]

    figure = Figure(figsize=(8, 2.3 * n_arms), layout="constrained")
    axes_column = figure.subplots(n_arms, 1, sharex=True, sharey=True)
    for arm, (axes, arm_state) in enumerate(zip(axes_column, snapshot.arms, strict=True)):
        colour = arm_colour(arm)
        axes.scatter(arm_state.contexts, arm_state.rewards, s=9, color=colour, alpha=0.35, linewidths=0)
        axes.fill_between(
            grid,
            arm_state.posterior_mean - z * arm_state.posterior_std,
            arm_state.posterior_mean + z * arm_state.posterior_std,
            color=colour,
            alpha=0.18,
            linewidth=0,
        )
        axes.plot(grid, arm_state.posterior_mean, color=colour)
        axes.plot(grid, result.true_mean_rewards[:, arm], color=INK, linestyle="--", linewidth=1.2)
        axes.set_title(f"Arm {arm}  ·  {len(arm_state.contexts)} observations in the model")
        axes.set_ylabel("Reward")
    axes_column[-1].set_xlabel("Context x")
    axes_column[0].set_xlim(grid[0], grid[-1])
    axes_column[0].set_ylim(-0.4, 1.4)

    handles = [
        Line2D([], [], color=INK, linestyle="--", linewidth=1.2, label="True mean reward"),
        Line2D([], [], color=MUTED_INK, label="Posterior mean"),
        Patch(color=MUTED_INK, alpha=0.25, label=f"{credibility:.0%} credible interval"),
        Line2D([], [], color=MUTED_INK, marker="o", linestyle="", alpha=0.5, label="Observed rewards"),
    ]
    figure.legend(handles=handles, loc="outside upper left", ncols=len(handles))
    return figure
