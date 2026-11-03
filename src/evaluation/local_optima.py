"""Local-optima behavior: run EM from many random starts and measure the spread.

EM only guarantees a local optimum. Random starts land in different basins —
some correct, some wrong — which is exactly why restarts and k-means++ matter.
"""

from __future__ import annotations

import numpy as np

from ..em.gmm import GaussianMixtureEM


def restart_spread(
    X: np.ndarray,
    n_components: int,
    *,
    n_restarts: int = 30,
    max_iter: int = 200,
    tol: float = 1e-6,
    init: str = "random",
) -> dict:
    """Final log-likelihood of each single-start fit, plus summary statistics."""
    X = np.asarray(X, dtype=float)
    finals = np.empty(n_restarts)
    for s in range(n_restarts):
        gmm = GaussianMixtureEM(
            n_components, n_init=1, init=init, max_iter=max_iter, tol=tol, random_state=s
        ).fit(X)
        finals[s] = gmm.log_likelihood_
    best = float(np.max(finals))
    worst = float(np.min(finals))
    return {
        "log_likelihoods": finals,
        "best": best,
        "worst": worst,
        "spread": best - worst,
        "suboptimal_fraction": float(np.mean(finals < best - 1e-3)),
    }
