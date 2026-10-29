# Contextual bandits with Gaussian-process rewards

Thompson sampling for a contextual bandit where each arm's reward is an unknown smooth function of a
one-dimensional context, modelled with one Gaussian process per arm. A sliding window over recent rounds lets
the agent recover when the reward functions change at unannounced change points.

* **Method and assumptions:** [`docs/methodology.md`](docs/methodology.md)
* **Results:** [`docs/report.html`](docs/report.html), a self-contained page with inline SVG figures that
  can be copied into a website as it is

## Usage

```bash
pip install -e ".[dev]"
```

```bash
python run.py
```

`run.py` runs the stationary and piecewise-stationary experiments (30 replications each, in parallel) and
writes `docs/report.html`. Use `--replications 4` for a quick run, `--workers N` to limit the number of
processes and `--output PATH` to write the report elsewhere.

```bash
pytest
```

## Layout

| Module | Responsibility |
|---|---|
| `bandit/environment.py` | Reward curves, change points, noisy rewards |
| `bandit/gaussian_process.py` | GP posterior over one arm's mean reward, with hyperparameters learned by maximum likelihood |
| `bandit/agent.py` | Thompson sampling over the per-arm GPs, with an optional sliding window |
| `bandit/simulation.py` | Experiment configuration, replications and summary statistics |
| `bandit/plotting.py` | Matplotlib figures |
| `bandit/report.py`, `bandit/report_template.html` | HTML report |

## Embedding the report

All report styles are scoped to `.cb-report`. To embed the report in another page, copy the
`<style id="cb-report-styles">` block and the `<article class="cb-report">` element; the
`standalone-page` styles only set the page background when the file is opened on its own. Dark mode
follows the reader's system setting, or a `data-theme="dark"`/`"light"` attribute on the root element.
