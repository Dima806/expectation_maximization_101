"""EM for missing data under a single multivariate Gaussian (MCAR).

The same guess-then-fit loop, pointed at missing values instead of hidden labels.
Treat the missing entries as the latent variables:

  E-step: fill each hole with its conditional expectation given the observed
          entries and the current (mean, covariance), accumulating the conditional
          covariance so the M-step covariance is not under-estimated.
  M-step: re-estimate (mean, covariance) from the completed sufficient statistics.

The observed-data log-likelihood is tracked and is monotone non-decreasing.
Unlike single mean-filling, this preserves the correlation structure.
"""

from __future__ import annotations

import numpy as np


def _observed_loglik(
    X: np.ndarray, observed: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> float:
    """Incomplete-data log-likelihood: sum of log N(x_obs; mean_obs, cov_obs)."""
    total = 0.0
    log_2pi = np.log(2.0 * np.pi)
    for i in range(X.shape[0]):
        o = observed[i]
        if not o.any():
            continue
        xo = X[i, o]
        mu = mean[o]
        cov_oo = cov[np.ix_(o, o)]
        _, logdet = np.linalg.slogdet(cov_oo)
        diff = xo - mu
        maha = float(diff @ np.linalg.inv(cov_oo) @ diff)
        total += -0.5 * (int(o.sum()) * log_2pi + logdet + maha)
    return total


def _impute(X: np.ndarray, observed: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fill missing entries with their conditional expectation under (mean, cov)."""
    out = X.copy()
    for i in range(X.shape[0]):
        o = observed[i]
        mi = ~o
        if not mi.any():
            continue
        if o.any():
            cov_oo_inv = np.linalg.inv(cov[np.ix_(o, o)])
            out[i, mi] = mean[mi] + cov[np.ix_(mi, o)] @ cov_oo_inv @ (X[i, o] - mean[o])
        else:
            out[i, mi] = mean[mi]
    return out


class GaussianMissingEM:
    """Estimate a Gaussian's mean/covariance from data with missing entries."""

    def __init__(
        self, *, max_iter: int = 200, tol: float = 1e-6, reg: float = 1e-6, random_state: int = 0
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> GaussianMissingEM:
        """Fit to ``X`` where missing entries are marked with NaN."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        observed = ~np.isnan(X)
        eye = np.eye(d)
        mean = np.nanmean(X, axis=0)
        filled = np.where(observed, X, mean)
        cov = np.cov(filled.T) + self.reg * eye
        history: list[float] = []
        for it in range(self.max_iter):
            history.append(_observed_loglik(X, observed, mean, cov))
            sum_x = np.zeros(d)
            sum_xx = np.zeros((d, d))
            for i in range(n):
                o = observed[i]
                mi = ~o
                xi = np.empty(d)
                xi[o] = X[i, o]
                extra = np.zeros((d, d))
                if mi.any():
                    if o.any():
                        cov_oo_inv = np.linalg.inv(cov[np.ix_(o, o)])
                        cov_mo = cov[np.ix_(mi, o)]
                        xi[mi] = mean[mi] + cov_mo @ cov_oo_inv @ (X[i, o] - mean[o])
                        cond_cov = cov[np.ix_(mi, mi)] - cov_mo @ cov_oo_inv @ cov_mo.T
                    else:
                        xi[mi] = mean[mi]
                        cond_cov = cov[np.ix_(mi, mi)]
                    extra[np.ix_(mi, mi)] = cond_cov
                sum_x += xi
                sum_xx += np.outer(xi, xi) + extra
            mean = sum_x / n
            cov = sum_xx / n - np.outer(mean, mean) + self.reg * eye
            if it > 0 and history[-1] - history[-2] < self.tol:
                break
        history.append(_observed_loglik(X, observed, mean, cov))
        self.mean_ = mean
        self.covariance_ = cov
        self.imputed_ = _impute(X, observed, mean, cov)
        self.loglik_history_ = np.asarray(history)
        self.log_likelihood_ = float(history[-1])
        return self
