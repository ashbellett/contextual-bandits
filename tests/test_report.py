import re

import numpy as np

from bandit.report import ReportSection, render_report
from bandit.simulation import ExperimentConfig, ExperimentResult, run_replication


def test_report_inlines_one_responsive_svg_per_figure_with_unique_ids():
    config = ExperimentConfig(name="tiny", horizon=20, replications=2, grid_size=20)
    seeds = np.random.SeedSequence(0).spawn(config.replications)
    result = ExperimentResult(config, [run_replication(config, seed) for seed in seeds])
    html = render_report([ReportSection("A", "first", result), ReportSection("B", "second", result)])

    assert html.count("<svg") == 6
    assert "<?xml" not in html
    assert re.search(r'<svg[^>]*width="100%"', html)
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    for reference in re.findall(r"url\(#([^)]+)\)", html):
        assert reference in ids
