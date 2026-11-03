"""Choosing the number of components with BIC / AIC.

EM fits the parameters for a fixed k; it does not tell you k. BIC and AIC score
the fitted log-likelihood against model complexity so you can compare ks.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..em.gmm import GaussianMixtureEM


def gmm_n_parameters(n_components: int, n_features: int, covariance_type: str = "full") -> int:
    """Free-parameter count of a Gaussian mixture (for BIC/AIC)."""
    k, d = n_components, n_features
    mean_params = k * d
    weight_params = k - 1
    cov_params = k * d * (d + 1) // 2 if covariance_type == "full" else k
    return mean_params + weight_params + cov_params


def bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Bayesian information criterion (lower is better)."""
    return -2.0 * log_likelihood + n_params * float(np.log(n_samples))


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike information criterion (lower is better)."""
    return -2.0 * log_likelihood + 2.0 * n_params


def select_n_components(
    X: np.ndarray,
    k_values: Iterable[int],
    *,
    covariance_type: str = "full",
    n_init: int = 5,
    max_iter: int = 200,
    random_state: int = 0,
) -> dict:
    """Fit a GMM for each k and report BIC/AIC plus the best k by each."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    results = []
    for k in k_values:
        gmm = GaussianMixtureEM(
            int(k),
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        ).fit(X)
        ll = gmm.log_likelihood_
        n_params = gmm_n_parameters(int(k), d, covariance_type)
        results.append(
            {
                "k": int(k),
                "log_likelihood": float(ll),
                "bic": bic(ll, n_params, n),
                "aic": aic(ll, n_params),
            }
        )
    best_bic_k = min(results, key=lambda r: r["bic"])["k"]
    best_aic_k = min(results, key=lambda r: r["aic"])["k"]
    return {"results": results, "best_bic_k": best_bic_k, "best_aic_k": best_aic_k}
