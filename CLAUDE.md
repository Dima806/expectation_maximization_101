# CLAUDE.md — expectation_maximization_101

Guidance for Claude Code (and humans) working in this repo. Read this before editing.

## Status

Scaffold + library + tests + 6 notebooks + Streamlit app are **built and green**:
`make ci` passes (sync · ruff format · ruff check · ty · 10 tests) and `make notebooks`
executes all six in ~45s total. The devcontainer and a CI workflow are intentionally **not**
present yet.

## Identity

Expectation-maximization **built from scratch on CPU**, as an educational "101" portfolio
project for data scientists. 6 notebooks, 1 Streamlit app, a typed `src/` library, tests.

**Thesis.** EM is a two-step loop — **E:** guess soft labels (responsibilities); **M:** fit the
model as if the guess were true — that *provably never lowers the log-likelihood*. It recovers
populations you never observed being mixed. **K-Means is hard EM** (responsibilities forced to
0/1, spherical equal covariance). **LCA is EM on categorical data** (mixture of multinomials).
The **same loop imputes missing data** (fill holes with expectations, refit, repeat) and
preserves the covariance that single mean-filling destroys.

**Build-up (the spine of the whole project):**
`chicken-and-egg → two coins → Gaussian mixture → K-Means is hard EM → missing data → when EM struggles`.
Every implementation is one variation on the same loop.

**Companion projects (cross-link, do not duplicate; verified live repos under
`github.com/Dima806`):** `clustering_arena` (GMM is EM), `categorical_clustering_arena`
(LCA is EM), `missing_data_imputation_arena` (EM is the engine under imputation),
`dim_reduction_arena` (EM extension: factor analysis / PPCA). This project is the *how it
works* to their *which wins* — add mechanism, not a competing benchmark.
(Note: the PRD also named `bayesian_101` and `imputation_arena`, which do **not** exist — do
not link them.)

## Stack & constraints

- **Python 3.11+** (the resolved venv here is 3.14). Package manager is **`uv` only — never
  `pip`, never a bare `python`/`pytest`.** Everything runs through `uv run ...` (the Makefile
  already does this).
- Runtime deps: `numpy`, `scipy`, `scikit-learn` (**validation/comparison only**, never in the
  from-scratch EM), `pandas`, `matplotlib`, `plotly`, `streamlit`, `pydantic`,
  `pydantic-settings`, `pyyaml`.
- Dev deps: `pytest`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
- **Hard budget: 2-CPU / 8 GB Codespace, no GPU.** All datasets are tiny (a few thousand points
  at most). **Every notebook must run in < 3 minutes.** If something is slow, the data is too big
  or the loop is wrong — do not reach for more compute.

## Commands

```
make setup       # first-time: install uv, uv sync --all-extras, register Jupyter kernel
make sync        # uv sync --all-extras
make lint        # format + check + typecheck (run before every commit)
make test        # uv run pytest
make notebooks   # execute all notebooks/0*.ipynb (fails if any errors; 180s timeout each)
make run         # streamlit app on :8501
make lab         # JupyterLab on :8888
make ci          # sync + lint + test   ·   make dev = lint + test (fast loop)
```

`make lint` and `make test` must pass with **zero errors** before anything is considered done.

## Structure

```
src/
  config.py              # pydantic-settings; loads config/settings.yaml (counts, inits, tols, seeds)
  em/
    two_coins.py         # canonical discrete EM — small enough to trace by hand
    gmm.py               # Gaussian mixture EM: e_step (responsibilities), M-step, log-likelihood
    kmeans.py            # K-Means as hard EM — kmeans_as_hard_em() drives the gmm scaffold
    latent_class.py      # LCA: mixture of multinomials (categorical EM)
    missing_data.py      # EM for Gaussian missing values (fill → refit → repeat)
  diagnostics/
    likelihood.py        # logsumexp + is_monotone_nondecreasing / assert_monotone
    init_strategies.py   # random_init, kmeans_plus_plus
    model_selection.py   # bic / aic / select_n_components
  datasets/
    mixtures.py          # PLANTED mixtures with known truth (returns truth dataclasses)
  evaluation/
    recovery.py          # match_components, parameter_recovery, clustering_accuracy, same_partition
    vs_sklearn.py        # from-scratch GMM vs sklearn.GaussianMixture (compare_gmm)
    local_optima.py      # restart_spread
  visualisation.py       # plot_loglik, plot_gmm_1d, plot_clusters_2d
notebooks/01..06         # see build-up above; executed with outputs embedded
app/streamlit_app.py     # EM animator, soft→hard slider, restart roulette, missing-data demo
tests/                   # one file per headline claim (see "Testing")
config/settings.yaml     # component counts, init strategies, tolerances, seeds
outputs/figures/*.png    # every notebook visual
outputs/data/*.json      # every notebook input, numerical result, and narrative caption
conftest.py              # root conftest so `import src` resolves in tests
```

The `em/` modules are the heart. Keep them **readable and from-scratch** — a reader should be
able to see the E-step and M-step as a few dozen lines. `kmeans.py`'s `kmeans_as_hard_em`
reuses `gmm.py`'s loop to make "K-Means is hard EM" literal in code, not just prose.

## Notebooks & outputs (the saving convention)

Every notebook starts with a shared preamble defining two helpers that **must** be used for
anything persisted:

- `savefig(fig, name)` → writes `outputs/figures/<name>.png` (150 dpi, tight bbox).
- `savejson(name, obj)` → writes `outputs/data/<name>.json` (numpy-aware encoder).

Rules when editing or adding notebooks:
- **Every visual is saved as a PNG** via `savefig`; **every textual/numerical input and result
  is saved as JSON** via `savejson`. Name files with the `nbNN_` prefix (e.g. `nb03_loglik`).
- Each notebook saves a `nbNN_inputs.json` (planted truth / knobs), result JSON(s), and a
  `nbNN_narrative.json` (title, E/M-step descriptions, figure captions).
- Paths resolve to the repo root regardless of cwd (the preamble walks up for `pyproject.toml`),
  so notebooks behave identically under `make notebooks` and interactive Jupyter.
- Keep each notebook < 3 min. Re-run with
  `uv run jupyter nbconvert --to notebook --execute --inplace <nb>` to refresh embedded outputs.

## Key concepts (get these right)

- **Chicken-and-egg:** labels → params and params → labels, but you have neither. Resolution:
  guess one, fit the other, iterate.
- **E-step:** responsibilities (prob each point belongs to each component). **M-step:** weighted
  re-estimation. Monotone log-likelihood is the headline invariant — **assert it** (a decrease
  is a bug).
- **K-Means = hard EM:** responsibilities forced to 0/1, spherical equal covariance, equal weights.
- **LCA:** EM on categorical data. **Missing data:** holes are the latent variables.
- **Failure modes (nb 06):** local optima (→ k-means++ + restarts), slow tail, component collapse
  (→ covariance regularization), choosing `k` (→ BIC/AIC).

## Numerical rules (non-negotiable)

- Use **log-sum-exp** (`src/diagnostics/likelihood.py`) — never multiply raw densities.
- **Regularize covariances** (`reg_covar`, a diagonal ridge) to avoid singular/collapsing
  components.
- **Seed everything** (data + init) so recovery is reproducible.
- `scikit-learn` appears **only** in `evaluation/vs_sklearn.py` and tests — never inside `em/`.

## Testing (asserted success criteria)

`test_two_coins` (recovers biases) · `test_gmm` (matches sklearn + recovers planted params) ·
`test_monotone` (log-likelihood never decreases: GMM, two-coins, missing-data) · `test_kmeans_em`
(hard-EM reproduces K-Means partition) · `test_missing_data` (EM beats mean-fill on covariance) ·
`test_latent_class` (recovers planted classes) · `test_model_selection` (BIC picks true `k`).
Keep tests tiny, seeded, and fast.

## Conventions & gotchas

- `ruff` line-length 99; import sorting on. **`N803`/`N806` are intentionally ignored** so
  math-convention names (`X`, `N`, `K`, `Soo`, …) are allowed — don't "fix" them to lowercase.
- Full type hints on public functions; `ty check src/` must be clean. ty's numpy stubs reject a
  few patterns: prefer `np.max(arr)` / `np.min(arr)` over `arr.max()` / `arr.min()` with no args,
  and don't annotate a function `-> np.ndarray` if a branch returns a Python `float`.
- The import package is literally **`src`** (hatch `packages = ["src"]`); `uv sync` installs it
  editable, and `conftest.py` keeps `import src` working in tests. Tests import via
  `from src.em.gmm import ...`.
- `*.nbconvert.ipynb` (left by `make notebooks`) is gitignored; `outputs/` is tracked.
- License is Apache-2.0.

## Token rules

- **Caveman mode: ON.** Code-first, minimal prose. Prefer diffs/snippets over essays.
- `/compact` at notebook boundaries.
- Don't restate this file back; just follow it.
