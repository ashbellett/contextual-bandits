# Contextual bandits with Gaussian-process rewards

This document describes the simulation in this repository: the environment, the agent, how regret is
measured and the assumptions behind each choice. The code mirrors the sections below:

| Section | Module |
|---|---|
| [Environment](#1-environment) | `bandit/environment.py` |
| [Regret](#2-regret) | `bandit/simulation.py` (`run_replication`) |
| [Reward model](#3-reward-model-one-gaussian-process-per-arm) | `bandit/gaussian_process.py` |
| [Thompson sampling](#4-thompson-sampling) and [sliding window](#5-non-stationarity-sliding-window) | `bandit/agent.py` |
| [Experiments](#6-experiments) | `run.py`, `bandit/simulation.py`, `bandit/report.py` |

## 1. Environment

There are $K$ arms and a one-dimensional context $x \in \mathcal{X} = [0, 1]$. At each round
$t = 0, 1, \dots, T-1$:

1. a context is drawn independently, $x_t \sim \mathrm{Uniform}(\mathcal{X})$;
2. the agent pulls an arm $a_t \in \{0, \dots, K-1\}$;
3. it observes $r_t = \mu_{a_t}^{(t)}(x_t) + \varepsilon_t$ with $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$, independent of everything else.

Only the pulled arm's reward is observed (bandit feedback).

**Reward curves.** Every mean-reward curve is built from one base function with a phase $\varphi$:

$$
g(x; \varphi) = u \sin u, \qquad u = 10x + \varphi \tfrac{\pi}{2}.
$$

This curve has several local maxima whose height grows with $u$, so the best arm changes several times
across the context interval. Each curve is rescaled to $[0, 1]$ over $\mathcal{X}$ so that all arms are on a
common scale and the noise level $\sigma$ means the same thing for every arm:

$$
\mu_a^{(t)}(x) = \frac{g(x; \varphi) - \min_{\mathcal{X}} g(\cdot\,; \varphi)}{\max_{\mathcal{X}} g(\cdot\,; \varphi) - \min_{\mathcal{X}} g(\cdot\,; \varphi)},
\qquad \varphi = a + K\,\rho(t).
$$

The minimum and maximum are taken over a dense grid of 2001 points.

**Change points.** With change points $\tau_1 < \tau_2 < \dots$, the *regime* $\rho(t)$ is the number of
change points at or before $t$. Inside a regime the environment is stationary. At a change point every arm's
phase moves on by $K$, so all curves change at once and the agent is not told. With
$K = 3$ the phases are $\{0, 1, 2\}$, then $\{3, 4, 5\}$, then $\{6, 7, 8\}$.

## 2. Regret

The regret at round $t$ is the expected reward lost by not pulling the best arm for the observed context:

$$
\Delta_t = \max_{a} \mu_a^{(t)}(x_t) - \mu_{a_t}^{(t)}(x_t) \;\ge\; 0,
\qquad R_T = \sum_{t=0}^{T-1} \Delta_t .
$$

This is the *pseudo-regret* against the best context-dependent policy, which is the quantity bandit
regret bounds are stated in. It uses the arm's mean reward $\mu_{a_t}(x_t)$, not the noisy observation
$r_t$. Subtracting $r_t$ would add zero-mean noise with variance $\sigma^2$ to every term, so regret could be
negative and the curves would be noisier, with no change in expectation. For non-stationary environments it
is the *dynamic* regret: the comparator is the best arm under the curves in force at round $t$.

Two reference points help to read $R_T$:

* **Uniform random policy.** Pulling arms uniformly at random has expected regret
  $\mathbb{E}_x\big[\max_a \mu_a(x) - \tfrac{1}{K}\sum_a \mu_a(x)\big]$ per round, which is linear in $T$.
  It is computed exactly on the context grid and reported in the HTML report.
* **Sub-linear regret** ($R_T / T \to 0$) means the agent eventually acts optimally almost all the time.
  This is the target in a stationary environment. When the environment keeps changing, no algorithm can
  achieve it over the whole horizon. The question becomes how quickly regret flattens again after each
  change.

## 3. Reward model: one Gaussian process per arm

The agent models each arm's mean-reward curve independently with a Gaussian process (GP):

$$
\mu_a(\cdot) \sim \mathcal{GP}\big(m, k_\theta\big), \qquad r = \mu_a(x) + \varepsilon,\; \varepsilon \sim \mathcal{N}(0, \sigma_n^2).
$$

Independent GPs per arm are the same as a single GP over (arm, context) pairs whose kernel is
$k\big((a, x), (a', x')\big) = \mathbb{1}[a = a']\,k_\theta(x, x')$, which is the setting of Krause & Ong
(2011) with no information shared between arms.

**Prior mean.** Rewards are normalised to $[0, 1]$, so the prior mean is the constant midpoint $m = 0.5$.
With a zero prior mean, arms with no nearby data would be biased towards looking bad.

**Kernel.** Signal variance times a Matérn kernel with smoothness $\nu = 5/2$, plus observation noise:

$$
k_\theta(x, x') = s^2 \Big(1 + \sqrt{5}\,d + \tfrac{5}{3}d^2\Big) e^{-\sqrt{5}\,d}, \quad d = \frac{|x - x'|}{\ell},
\qquad \theta = (s^2, \ell, \sigma_n^2).
$$

Matérn-5/2 functions are twice differentiable. This is smooth enough for the reward curves above and less
restrictive than the infinitely smooth squared-exponential kernel.

**Posterior.** Given $n$ observations $(X, \mathbf{y})$ of one arm, the posterior of the *mean reward* at a
context $x$ is Gaussian with

$$
\mu_n(x) = m + \mathbf{k}(x)^\top (K + \sigma_n^2 I)^{-1}(\mathbf{y} - m\mathbf{1}), \qquad
\sigma_n^2(x) = k(x, x) - \mathbf{k}(x)^\top (K + \sigma_n^2 I)^{-1}\mathbf{k}(x),
$$

where $K_{ij} = k_\theta(x_i, x_j)$ and $\mathbf{k}(x)_i = k_\theta(x, x_i)$ (Rasmussen & Williams, 2006,
§2.2). This is the posterior of the latent curve $\mu_a(x)$. The predictive for a new *observation* would add
$\sigma_n^2$ to the variance. The agent needs the latent posterior because it reasons about mean rewards.
scikit-learn's `predict` returns the observation predictive when the kernel contains a `WhiteKernel`, so
`GaussianProcessModel.predict` subtracts the learned noise variance.

**Hyperparameters (type-II maximum likelihood).** $\theta$ is chosen by maximising the log marginal
likelihood of the arm's observations,

$$
\log p(\mathbf{y} \mid X, \theta) = -\tfrac12 (\mathbf{y} - m\mathbf{1})^\top (K + \sigma_n^2 I)^{-1} (\mathbf{y} - m\mathbf{1})
- \tfrac12 \log\lvert K + \sigma_n^2 I\rvert - \tfrac{n}{2}\log 2\pi,
$$

with L-BFGS-B inside the bounds $s^2 \in [10^{-3}, 10]$, $\ell \in [0.01, 1]$ and
$\sigma_n^2 \in [10^{-5}, 1]$. Each optimisation starts from the previous estimate. Optimising after every
observation costs $O(n^3)$ per likelihood evaluation, so it would dominate the run time. Instead:

* hyperparameters are re-optimised every 25 pulls of an arm (`optimise_every`);
* they are not optimised until the arm has 10 observations in its window, because maximum likelihood on a
  handful of points is unreliable. Until then the initial values $s^2 = 0.1$, $\ell = 0.2$,
  $\sigma_n^2 = 0.1$ are used;
* between optimisations, the posterior is still updated exactly with every new observation, holding
  $\theta$ fixed.

Learning $\theta$ from the data the agent itself collected makes the analysis data-dependent. The regret
guarantees cited below assume a known kernel and noise level. In practice, maximum likelihood recovers the
true noise variance well here ($\hat\sigma_n^2 \approx 0.01$ for $\sigma = 0.1$; see
`tests/test_gaussian_process.py`).

## 4. Thompson sampling

At each round the agent draws one plausible mean reward per arm from its posterior and acts greedily with
respect to the draw:

```
for t = 0, 1, ..., T-1:
    observe context x_t
    for each arm a:  draw  μ̃_a ~ N( μ_{a,t}(x_t), σ²_{a,t}(x_t) )     # posterior given data before t
    pull a_t = argmax_a μ̃_a and observe r_t
    add (t, x_t, r_t) to arm a_t's history
```

Only the posterior *marginal* at the current context matters for the decision, so sampling the whole curve
is unnecessary. Each draw is a single univariate normal, which is the exact marginal of a full posterior
sample path at $x_t$. Arms that are uncertain at $x_t$ have wide posteriors and occasionally produce the
largest draw. This is how Thompson sampling explores, and it explores less as the posteriors concentrate
(Russo et al., 2018).

Draws must be independent across rounds and arms. Every draw comes from a single `numpy.random.Generator`
owned by the agent and seeded per replication. Reusing a fixed seed for every draw makes the "sample" a
deterministic $\mu + c\,\sigma$ with the same $c$ every time. That is an upper-confidence-bound rule, not
Thompson sampling. See [Appendix A](#appendix-a-review-of-the-original-implementation).

GP Thompson sampling enjoys $\tilde{O}(\gamma_T \sqrt{T})$ regret for a known kernel, where $\gamma_T$ is
the maximum information gain of the kernel (Chowdhury & Gopalan, 2017). That analysis inflates the posterior
variance by a slowly growing factor. As is common in practice, the agent here samples from the unscaled
posterior.

## 5. Non-stationarity: sliding window

A stationary GP conditions on every observation forever. After a change point its old observations
contradict the new curves, and the posterior can stay confidently wrong for a long time. The agent therefore
only conditions on observations from the most recent $W$ **rounds**:

$$
\mathcal{D}_{a,t} = \{(x_s, r_s) : a_s = a,\; t - W \le s < t\}.
$$

This is the sliding-window idea of Garivier & Moulines (SW-UCB), carried over to GPs as in Zhou & Shroff
(2021). The window is measured in rounds, not in each arm's own pulls, and that matters:

* if each arm kept its last $W$ *pulls*, an arm that looked bad before a change point and stopped being
  pulled would keep its stale, confident posterior indefinitely and might never be tried again, even if it
  became the best arm;
* with a window in rounds, an arm that has not been pulled for $W$ rounds has no data left. Its posterior
  reverts to the prior, its draws become wide again, and it is re-explored.

$W$ controls a bias–variance trade-off. A short window adapts quickly after a change but always works with
little data, so it keeps paying some exploration regret even when nothing changes. A long window learns
the curves well but takes longer to forget. For SW-UCB with $\Upsilon_T$ change points the window that
balances the two is of order $\sqrt{T \log T / \Upsilon_T}$. Here $W = 200$ with change points every 1000
rounds.

When the window is turned off (`window=None`) the agent is the ordinary stationary GP-TS agent.

## 6. Experiments

| | Stationary | Piecewise-stationary |
|---|---|---|
| Arms $K$ | 3 | 3 |
| Rounds $T$ | 1000 | 3000 |
| Change points | none | 1000, 2000 |
| Noise $\sigma$ | 0.1 | 0.1 |
| Window $W$ | none | 200 rounds |
| Hyperparameter refit | every 25 pulls | every 25 pulls |
| Replications | 30 | 30 |

Each replication is an independent episode with its own environment noise, contexts and Thompson draws.
Seeds come from `numpy.random.SeedSequence(seed).spawn(replications)`, so results are reproducible and
replications are statistically independent. They run in parallel processes, each limited to one BLAS
thread.

The report shows, for each experiment:

* **Cumulative regret** $R_t$ averaged over replications, with a 95% confidence interval for the mean
  ($\pm 1.96\, \hat{s} / \sqrt{n_{\text{rep}}}$).
* **Best-arm map**: for each context in the final regime, the true optimal arm against the arm with the
  highest posterior mean (posterior means averaged over replications). The headline accuracy is the share
  of (replication, context) pairs where the posterior-mean argmax is the optimal arm.
* **Posterior plots** for one replication: each arm's true curve, the observations its GP is conditioned on
  at the end (the final window when one is used), the posterior mean and a 95% credible band for
  $\mu_a(x)$.

Because Thompson sampling pulls an arm mainly where it looks best, each GP is accurate where its arm is
competitive and stays uncertain elsewhere. That is the intended behaviour, not a modelling failure.

## 7. Limitations and extensions

* **One-dimensional contexts.** The GP code works with vectors, but the reward curves and figures assume
  $x \in [0, 1]$. Moving to $\mathbb{R}^d$ would need a $d$-dimensional reward curve (for example $g$ applied
  to $\lVert x \rVert$) and an anisotropic kernel.
* **Cost.** Exact GP inference is $O(n^3)$ in the number of observations per arm. The sliding window bounds
  $n$ by $W$; long stationary runs would need sparse or incremental GP updates.
* **Choosing $W$** requires a rough idea of how often the environment changes. Alternatives that avoid this
  are change-point detection with restarts (Besson et al., 2022), time-varying kernels that discount old
  observations smoothly (Bogunovic et al., 2016), or a GP over (context, time).
* **Shared structure.** Arms do not share information. A coregionalised kernel over arms could learn
  faster when arms' curves are related.

## Appendix A. Review of the original implementation

The first version of this simulation had the following problems, all fixed in the current code.

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | `GaussianProcessRegressor.sample_y` was called without `random_state`, so it always used seed 0. | Every "Thompson sample" was $\mu + 1.76\,\sigma$: a fixed-coefficient UCB rule, not Thompson sampling. | Draw $\tilde\mu_a \sim \mathcal{N}(\mu, \sigma^2)$ with the agent's own seeded `Generator`. |
| 2 | `Reward.reset()` replaced the configured GP (Matérn, `alpha=0.5`) with `GaussianProcessRegressor()`, and `reset()` ran before every replication. | The configured model was never used. Every run used a squared-exponential kernel with almost no noise term (`alpha=1e-10`), interpolating noisy data. Depending on the scikit-learn version its length scale was either fixed at 1 or collapsed towards 0. | One explicitly configured GP per arm, created fresh for every replication. |
| 3 | `alpha=0.5` was the assumed noise *variance*, while the true noise standard deviation was 0.1 (variance 0.01). | Even if used, the model would have assumed 50× too much noise and over-smoothed. | Noise variance is learned by maximum likelihood (`WhiteKernel`). |
| 4 | Regret subtracted the noisy observed reward. | Regret could be negative and was noisier, with the same expectation. | Pseudo-regret on mean rewards (§2). |
| 5 | At a change point the arms were replaced with shifted copies, and the shifted copies were carried into the next replication, whose `reset()` cleared data but not the shift. | Replications 2–30 of the dynamic experiment spent rounds 0–999 in the *final* regime, so the first regime appeared only once. | The environment is an immutable object; regimes are a function of $t$. |
| 6 | The window kept each arm's last $W$ pulls. | Rarely pulled arms kept stale data after a change and might never be re-explored. | Window measured in rounds (§5). |
| 7 | Contexts were reshaped with `reshape(-1, 1)`, which turns a $d$-vector into $d$ separate one-dimensional points. | `dimension > 1` silently produced wrong results. | Contexts are explicitly one-dimensional (§7). |
| 8 | `reward_variance` was passed as the normal distribution's `scale`. | The parameter was a standard deviation named as a variance. | Renamed `noise_std`. |
| 9 | Global `np.random.seed`; replications ran sequentially and shared one random stream. | Results depended on execution order; slow. | Per-replication `SeedSequence` streams; parallel processes. |

Smaller clean-ups: unused state (`true_maximiser`, `true_maximum`, `play_count`, the unused `plot_3d`
module), three copies of the domain constant, and `run.py` duplicating the plotting code for each experiment.
The notebook text described a context interval of $[0, 10]$ while the code used $[0, 1]$, and described
Thompson sampling as choosing the arm with the highest *expected* reward rather than the highest *sampled*
reward.

## References

1. Agrawal S., Goyal N. Thompson sampling for contextual bandits with linear payoffs. *International Conference on Machine Learning*. 2013.
2. Besbes O., Gur Y., Zeevi A. Stochastic multi-armed bandit problem with non-stationary rewards. *Advances in Neural Information Processing Systems*. 2014. 27.
3. Besson L., Kaufmann E., Maillard O., Seznec J. Efficient change-point detection for tackling piecewise-stationary bandits. *Journal of Machine Learning Research*. 2022. 23(77):1–40.
4. Bogunovic I., Scarlett J., Cevher V. Time-varying Gaussian process bandit optimization. *International Conference on Artificial Intelligence and Statistics*. 2016. 51:314–323.
5. Chowdhury S. R., Gopalan A. On kernelized multi-armed bandits. *International Conference on Machine Learning*. 2017.
6. Garivier A., Moulines E. On upper-confidence bound policies for switching bandit problems. *Algorithmic Learning Theory*. 2011. (arXiv:0805.3415, 2008.)
7. Krause A., Ong C. Contextual Gaussian process bandit optimization. *Advances in Neural Information Processing Systems*. 2011. 24.
8. Rasmussen C. E., Williams C. K. I. *Gaussian Processes for Machine Learning*. MIT Press. 2006.
9. Russo D., Van Roy B., Kazerouni A., Osband I., Wen Z. A tutorial on Thompson sampling. *Foundations and Trends in Machine Learning*. 2018. 11(1):1–96.
10. Srinivas N., Krause A., Kakade S. M., Seeger M. Gaussian process optimization in the bandit setting: no regret and experimental design. *International Conference on Machine Learning*. 2010.
11. Wu Q., Iyer N., Wang H. Learning contextual bandits in a non-stationary environment. *International ACM SIGIR Conference on Research & Development in Information Retrieval*. 2018. 41:495–504.
12. Zhou X., Shroff N. No-regret algorithms for time-varying Bayesian optimization. *Conference on Information Sciences and Systems*. 2021.
