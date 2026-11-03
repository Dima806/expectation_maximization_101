# expectation_maximization_101

**The algorithm that learns from data it cannot see.**

A hands-on, from-scratch introduction to **expectation-maximization** — the algorithm
hiding inside Gaussian mixtures, latent class analysis, hidden Markov models, and half
the missing-data methods you have ever used. Everything runs on CPU inside a 2-CPU / 8 GB
GitHub Codespace, with no GPU.

> EM is a two-step loop — **E:** guess the hidden labels (soft); **M:** fit the model as if
> the guess were true — that *provably never lowers the likelihood*. It recovers populations
> you never observed being mixed. **K-Means is this loop with hard assignments.** **LCA is
> this loop on categorical data.** **Missing-data imputation is this loop pointed at the
> holes.** The thing that feels like magic is a few dozen lines you can write yourself.

---

## What's inside

- **`src/` — a small, typed, from-scratch library** (no `scikit-learn` in the EM core;
  sklearn appears only as a validation oracle).
- **6 notebooks** that build the idea up one step at a time.
- **A Streamlit app** with four interactive demos.
- **A test suite** that asserts every headline claim (parameter recovery, monotone
  log-likelihood, K-Means ≡ hard EM, EM imputation beats mean-fill, BIC picks the true `k`).
- **`outputs/`** — every notebook figure saved as a **PNG**, and every input/result/caption
  saved as **JSON**, ready to reuse downstream.

## The six notebooks

| # | Notebook | What it shows |
|---|----------|---------------|
| 01 | [The Chicken and the Egg](notebooks/01_the_chicken_and_egg.ipynb) | The bind (labels ⇄ parameters, you have neither) and the bootstrap that breaks it. |
| 02 | [Two Coins](notebooks/02_two_coins.ipynb) | EM at its most transparent — E-step and M-step traced by hand on flip-counts. |
| 03 | [Gaussian Mixture From Scratch](notebooks/03_gaussian_mixture_from_scratch.ipynb) | **The showpiece.** Components separating out of the tangle; monotone log-likelihood; validated vs sklearn. |
| 04 | [K-Means Is Hard EM](notebooks/04_kmeans_is_hard_em.ipynb) | Force the responsibilities to all-or-nothing and the same loop *is* K-Means. |
| 05 | [Missing Data](notebooks/05_missing_data.ipynb) | The same loop imputes missing values, preserving correlations mean-filling destroys. |
| 06 | [When EM Struggles](notebooks/06_when_em_struggles.ipynb) | Local optima, k-means++ restarts, BIC/AIC model selection, and a decision framework. |

## Quickstart

```bash
make setup       # install uv (if needed), sync deps, register the Jupyter kernel
make test        # run the assertion suite (should be all green)
make notebooks   # execute all 6 notebooks (~45s total) → regenerates outputs/
make run         # launch the Streamlit app on :8501
make lab         # JupyterLab on :8888
```

Common developer loop:

```bash
make lint        # ruff format + ruff check --fix + ty typecheck
make ci          # sync + lint + test  (what CI runs)
```

Requires Python 3.11+. All dependencies are managed with [`uv`](https://docs.astral.sh/uv/)
— there is no `pip` step and no GPU.

## Repository layout

```
src/
  em/            two_coins · gmm · kmeans (hard EM) · latent_class (LCA) · missing_data
  diagnostics/   likelihood (log-sum-exp + monotonicity) · init_strategies (k-means++) · model_selection (BIC/AIC)
  datasets/      mixtures — planted truth for every generator (Gaussian, two-coins, categorical, with-holes)
  evaluation/    recovery (vs planted truth) · vs_sklearn · local_optima (restart spread)
  visualisation.py
notebooks/       01 … 06
app/             streamlit_app.py — EM animator · soft-to-hard slider · restart roulette · missing-data demo
tests/           one test file per headline claim
config/          settings.yaml  (component counts, init strategies, tolerances, seeds)
outputs/
  figures/       <nb>_*.png   — every visual
  data/          <nb>_*.json  — every input, numerical result, and narrative caption
```

## Design principles

- **From scratch.** The E-step, M-step, and log-likelihood are written out in plain NumPy.
  `scikit-learn` is used only to *check* the from-scratch GMM, never inside it.
- **Planted truth everywhere.** Every dataset is synthetic with known generating parameters,
  because recovery is only meaningful when you can score it against the truth that was hidden.
- **Numerically honest.** Log-sum-exp for stability, covariance regularization against
  collapse, seeded data and initializations for reproducibility.
- **Asserted, not asserted-by-vibes.** The monotone log-likelihood, sklearn agreement,
  K-Means equivalence, and imputation advantage are all checked in `tests/`.

## Verified results (from the notebooks / tests)

- Two-coins EM recovers planted biases `0.80 / 0.35` as `≈ 0.81 / 0.37`.
- From-scratch GMM matches `sklearn.GaussianMixture` average log-likelihood to `~1e-6`.
- Hard-EM path reproduces K-Means **assignment for assignment** (identical partition).
- On unequal-spread clusters, soft GMM accuracy `≈ 0.96` vs K-Means `≈ 0.80`.
- EM imputation recovers correlation `0.85 → ≈ 0.86` where mean-fill collapses it to `≈ 0.63`.
- BIC recovers the true component count (`k = 3`) on well-separated data.

## Related projects

Part of a wider series of hands-on data-science projects by [Dima806](https://github.com/Dima806).
EM is the engine underneath several of them:

- **[clustering_arena](https://github.com/Dima806/clustering_arena)** — a head-to-head
  comparison of clustering algorithms on real-world datasets. The Gaussian mixture that wins
  there *is* EM; this project opens it up.
- **[categorical_clustering_arena](https://github.com/Dima806/categorical_clustering_arena)** —
  a comparison of categorical clustering methods. Latent class analysis there is EM on
  categorical data (a mixture of multinomials).
- **[missing_data_imputation_arena](https://github.com/Dima806/missing_data_imputation_arena)** —
  a comparison of missing-data imputation strategies. EM is the principled engine under proper
  imputation (notebook 05).
- **[dim_reduction_arena](https://github.com/Dima806/dim_reduction_arena)** — a comparison of
  dimensionality-reduction methods. A natural EM extension (factor analysis / probabilistic PCA)
  lives here.

## References

- Dempster, Laird & Rubin (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm*, JRSS B. — the foundational paper.
- Do & Batzoglou (2008), *What is the Expectation Maximization Algorithm?*, Nature Biotechnology. — the two-coins exposition (notebook 02).
- Bishop (2006), *Pattern Recognition and Machine Learning*, Ch. 9.
- McLachlan & Krishnan (2008), *The EM Algorithm and Extensions*.
- Little & Rubin (2019), *Statistical Analysis with Missing Data*.

## License

Apache-2.0. See [LICENSE](LICENSE).
