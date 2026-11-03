"""Gaussian mixture EM, from scratch (the showpiece).

E-step: responsibilities r_ik = P(point i belongs to component k | params), via
log-sum-exp for stability. M-step: weighted re-estimation of each component's
weight, mean, and covariance. The incomplete-data log-likelihood is tracked and
is monotone non-decreasing every iteration.

The ``hard``, ``covariance_type="spherical"``, ``tied`` and ``update_weights``
flags exist so the very same loop reduces to K-Means (see em/kmeans.py).
"""

from __future__ import annotations

import numpy as np

from ..diagnostics.init_strategies import kmeans_plus_plus, random_init
from ..diagnostics.likelihood import logsumexp


def _log_gaussian(X: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Log multivariate-normal density of each point under each component, (N, K)."""
    n, d = X.shape
    k = means.shape[0]
    out = np.empty((n, k))
    const = d * np.log(2.0 * np.pi)
    for j in range(k):
        cov = covariances[j]
        _, logdet = np.linalg.slogdet(cov)
        prec = np.linalg.inv(cov)
        diff = X - means[j]
        maha = np.einsum("ij,jk,ik->i", diff, prec, diff)
        out[:, j] = -0.5 * (const + logdet + maha)
    return out


def e_step(
    X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return log-responsibilities (N, K) and the total log-likelihood."""
    log_prob = _log_gaussian(X, means, covariances)
    weighted = log_prob + np.log(np.clip(weights, 1e-300, None))
    log_norm = logsumexp(weighted, axis=1)
    log_resp = weighted - log_norm[:, None]
    return log_resp, float(log_norm.sum())


class GaussianMixtureEM:
    """From-scratch Gaussian mixture via expectation-maximization."""

    def __init__(
        self,
        n_components: int,
        *,
        covariance_type: str = "full",
        tied: bool = False,
        hard: bool = False,
        update_weights: bool = True,
        init: str = "kmeans++",
        max_iter: int = 200,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tied = tied
        self.hard = hard
        self.update_weights = update_weights
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

    def _init_means(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.init == "random":
            return random_init(X, self.n_components, rng)
        return kmeans_plus_plus(X, self.n_components, rng)

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, d = X.shape
        k = self.n_components
        nk = resp.sum(axis=0) + 1e-12
        means = (resp.T @ X) / nk[:, None]
        eye = np.eye(d)
        if self.covariance_type == "full":
            covariances = np.empty((k, d, d))
            for j in range(k):
                diff = X - means[j]
                cov = (resp[:, j : j + 1] * diff).T @ diff / nk[j]
                covariances[j] = cov + self.reg_covar * eye
            return means, covariances
        # spherical: one scalar variance per component (optionally tied across all)
        var = np.empty(k)
        for j in range(k):
            diff = X - means[j]
            var[j] = (resp[:, j] * (diff**2).sum(axis=1)).sum() / (nk[j] * d)
        if self.tied:
            total = 0.0
            for j in range(k):
                diff = X - means[j]
                total += float((resp[:, j] * (diff**2).sum(axis=1)).sum())
            var[:] = total / (n * d)
        var = var + self.reg_covar
        covariances = np.stack([eye * var[j] for j in range(k)])
        return means, covariances

    def _fit_single(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, d = X.shape
        k = self.n_components
        means = self._init_means(X, rng)
        global_var = float(X.var(axis=0).mean()) + self.reg_covar
        covariances = np.stack([np.eye(d) * global_var for _ in range(k)])
        weights = np.full(k, 1.0 / k)
        history: list[float] = []
        for it in range(self.max_iter):
            log_resp, ll = e_step(X, weights, means, covariances)
            history.append(ll)
            resp = np.exp(log_resp)
            if self.hard:
                hard_resp = np.zeros_like(resp)
                hard_resp[np.arange(n), resp.argmax(axis=1)] = 1.0
                resp = hard_resp
            if self.update_weights:
                weights = resp.sum(axis=0) / n
            means, covariances = self._m_step(X, resp)
            if it > 0 and ll - history[-2] < self.tol:
                break
        _, final_ll = e_step(X, weights, means, covariances)
        history.append(final_ll)
        return weights, means, covariances, np.asarray(history)

    def fit(self, X: np.ndarray) -> GaussianMixtureEM:
        """Fit via ``n_init`` restarts, keeping the highest-likelihood solution."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for _ in range(self.n_init):
            weights, means, covariances, history = self._fit_single(X, rng)
            if best is None or history[-1] > best[0]:
                best = (float(history[-1]), weights, means, covariances, history)
        assert best is not None
        _, weights, means, covariances, history = best
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.loglik_history_ = history
        self.log_likelihood_ = float(history[-1])
        self.n_iter_ = len(history) - 1
        log_resp, _ = e_step(X, weights, means, covariances)
        self.responsibilities_ = np.exp(log_resp)
        self.labels_ = self.responsibilities_.argmax(axis=1)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft responsibilities for new data, (N, K)."""
        X = np.asarray(X, dtype=float)
        log_resp, _ = e_step(X, self.weights_, self.means_, self.covariances_)
        return np.exp(log_resp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard component assignment (argmax responsibility)."""
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X: np.ndarray) -> float:
        """Average log-likelihood per sample."""
        X = np.asarray(X, dtype=float)
        _, ll = e_step(X, self.weights_, self.means_, self.covariances_)
        return ll / X.shape[0]
