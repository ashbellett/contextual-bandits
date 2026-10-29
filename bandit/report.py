"""Self-contained HTML report with inline SVG figures, ready to drop into a website."""

import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from html import escape
from importlib.resources import files
from string import Template

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure
from scipy.stats import norm

from bandit.plotting import STYLE, plot_arm_posteriors, plot_best_arm, plot_cumulative_regret
from bandit.simulation import ExperimentResult


@dataclass(frozen=True)
class ReportSection:
    title: str
    description: str
    result: ExperimentResult


def figure_to_svg(figure: Figure, id_prefix: str) -> str:
    """Inline-ready SVG: responsive width, no XML prolog, ids unique within the page."""
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"\sheight="[^"]*"', r'\1 width="100%"', svg, count=1)
    svg = re.sub(r'\bid="([^"]+)"', rf'id="{id_prefix}-\1"', svg)
    svg = re.sub(r'href="#([^"]+)"', rf'href="#{id_prefix}-\1"', svg)
    return re.sub(r"url\(#([^)]+)\)", rf"url(#{id_prefix}-\1)", svg)


def render_report(sections: list[ReportSection]) -> str:
    body = "\n".join(_render_section(index, section) for index, section in enumerate(sections))
    template = files("bandit").joinpath("report_template.html").read_text(encoding="utf-8")
    return Template(template).substitute(body=body)


def _render_section(index: int, section: ReportSection) -> str:
    result = section.result
    config = result.config
    plots: list[tuple[Callable[[ExperimentResult], Figure], str]] = [
        (
            plot_cumulative_regret,
            f"Cumulative regret averaged over {config.replications} replications; the band is a "
            "95% confidence interval for the mean.",
        ),
        (
            plot_best_arm,
            "Optimal arm at each context in the final regime, compared with the arm whose posterior "
            "mean (averaged over replications) is highest.",
        ),
        (
            plot_arm_posteriors,
            "Posterior of each arm's mean reward at the end of one replication, with the observations "
            "it is conditioned on. Arms are sampled where they look best, so the posterior is tight "
            "there and wide elsewhere.",
        ),
    ]
    figures = []
    for plot_index, (plot, caption) in enumerate(plots):
        with rc_context(STYLE):
            svg = figure_to_svg(plot(result), id_prefix=f"s{index}f{plot_index}")
        svg = svg.replace("<svg", f'<svg role="img" aria-label="{escape(caption)}"', 1)
        figures.append(f"<figure>{svg}<figcaption>{escape(caption)}</figcaption></figure>")

    return SECTION.format(
        title=escape(section.title),
        description=escape(section.description),
        stats=_render_stats(result),
        settings=_render_settings(result),
        figures="\n".join(figures),
    )


def _render_stats(result: ExperimentResult) -> str:
    final = result.cumulative_regret[:, -1]
    half_width = norm.ppf(0.975) * final.std(ddof=1) / np.sqrt(len(final))
    tail = max(1, result.config.horizon // 10)
    tail_regret = np.mean([r.regret[-tail:].mean() for r in result.replications])
    stats = [
        (f"{final.mean():.1f} ± {half_width:.1f}", f"Cumulative regret after {result.config.horizon} rounds"),
        (
            f"{result.uniform_policy_cumulative_regret[-1]:.0f}",
            "Expected regret of a uniformly random policy",
        ),
        (f"{tail_regret:.3f}", f"Mean regret per round over the last {tail} rounds"),
        (f"{result.best_arm_accuracy:.0%}", "Contexts where the learned best arm is optimal"),
    ]
    return "\n".join(
        f'<div class="stat"><span class="value">{escape(value)}</span>'
        f'<span class="label">{escape(label)}</span></div>'
        for value, label in stats
    )


def _render_settings(result: ExperimentResult) -> str:
    config = result.config
    settings = [
        ("Arms", config.n_arms),
        ("Rounds", config.horizon),
        ("Change points", ", ".join(map(str, config.change_points)) or "none"),
        ("Reward noise σ", config.noise_std),
        ("Sliding window", f"{config.window} rounds" if config.window else "none (all history)"),
        ("Hyperparameter refit", f"every {config.optimise_every} pulls of an arm"),
        ("Replications", config.replications),
    ]
    return "\n".join(f"<dt>{escape(key)}</dt><dd>{escape(str(value))}</dd>" for key, value in settings)


SECTION = """
<section>
  <h2>{title}</h2>
  <p>{description}</p>
  <div class="stats">
{stats}
  </div>
  <dl class="settings">
{settings}
  </dl>
{figures}
</section>
"""
