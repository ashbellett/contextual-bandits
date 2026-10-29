"""Run the static and dynamic experiments and write the HTML report."""

import argparse
import time
from dataclasses import replace
from pathlib import Path

from bandit.report import ReportSection, render_report
from bandit.simulation import ExperimentConfig, run_experiment

STATIC = ExperimentConfig(name="static", horizon=1000)
DYNAMIC = ExperimentConfig(name="dynamic", horizon=3000, change_points=(1000, 2000), window=200)

EXPERIMENTS = (
    (
        STATIC,
        "Stationary environment",
        "The reward curves never change and every observation is kept. Regret should grow quickly while "
        "the agent explores and then flatten out (sub-linear growth) once it has learned which arm is best "
        "where.",
    ),
    (
        DYNAMIC,
        "Piecewise-stationary environment",
        f"Every arm's reward curve is replaced at rounds {' and '.join(map(str, DYNAMIC.change_points))}. "
        f"The agent only conditions on the last {DYNAMIC.window} rounds, so it forgets the old curves and "
        "recovers after each change. Forgetting also costs a little regret between changes, so cumulative "
        "regret keeps growing slowly instead of flattening.",
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, help="override the number of replications")
    parser.add_argument("--workers", type=int, help="parallel processes (default: all CPUs)")
    parser.add_argument("--output", type=Path, default=Path("docs/report.html"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    sections = []
    for config, title, description in EXPERIMENTS:
        if arguments.replications:
            config = replace(config, replications=arguments.replications)
        start = time.perf_counter()
        result = run_experiment(config, workers=arguments.workers)
        print(f"{config.name}: {config.replications} replications in {time.perf_counter() - start:.0f} s")
        sections.append(ReportSection(title, description, result))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_report(sections), encoding="utf-8")
    print(f"Report written to {arguments.output}")


if __name__ == "__main__":
    main()
