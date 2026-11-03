"""Latent class analysis: EM on categorical data (mixture of multinomials).

The categorical twin of the Gaussian mixture. Each hidden class has, per feature,
a categorical distribution over response options. The E-step computes class
responsibilities; the M-step re-estimates each class's per-feature category
probabilities as responsibility-weighted frequencies.
"""

from __future__ import annotations

import numpy as np

from ..diagnostics.likelihood import logsumexp


def _log_categorical(X: np.ndarray, class_probs: np.ndarray) -> np.ndarray:
    """Log P(x_i | class k) for each point and class, (N, K)."""
    n, m = X.shape
    k = class_probs.shape[0]
    log_probs = np.log(np.clip(class_probs, 1e-12, None))  # (k, m, c)
    out = np.zeros((n, k))
    for j in range(m):
        out += log_probs[:, j, :][:, X[:, j]].T  # (k, n) -> (n, k)
    return out


class LatentClassEM:
    """EM for a mixture of ``n_classes`` categorical response profiles."""

    def __init__(
        self,
        n_classes: int,
        n_categories: int,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_init: int = 5,
        random_state: int = 0,
    ) -> None:
        self.n_classes = n_classes
        self.n_categories = n_categories
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def _fit_single(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, m = X.shape
        k, c = self.n_classes, self.n_categories
        class_probs = rng.dirichlet(np.ones(c), size=(k, m))  # (k, m, c)
        weights = np.full(k, 1.0 / k)
        history: list[float] = []
        for it in range(self.max_iter):
            log_lik = _log_categorical(X, class_probs)
            weighted = log_lik + np.log(np.clip(weights, 1e-300, None))
            log_norm = logsumexp(weighted, axis=1)
            history.append(float(log_norm.sum()))
            resp = np.exp(weighted - log_norm[:, None])
            weights = resp.sum(axis=0) / n
            new_probs = np.empty((k, m, c))
            for j in range(m):
                for cat in range(c):
                    new_probs[:, j, cat] = resp[X[:, j] == cat].sum(axis=0)
            new_probs += 1e-6
            new_probs /= new_probs.sum(axis=2, keepdims=True)
            class_probs = new_probs
            if it > 0 and history[-1] - history[-2] < self.tol:
                break
        log_lik = _log_categorical(X, class_probs)
        weighted = log_lik + np.log(np.clip(weights, 1e-300, None))
        log_norm = logsumexp(weighted, axis=1)
        history.append(float(log_norm.sum()))
        resp = np.exp(weighted - log_norm[:, None])
        return weights, class_probs, resp, np.asarray(history)

    def fit(self, X: np.ndarray) -> LatentClassEM:
        """Fit via ``n_init`` restarts, keeping the highest-likelihood solution."""
        X = np.asarray(X, dtype=int)
        rng = np.random.default_rng(self.random_state)
        best: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for _ in range(self.n_init):
            weights, class_probs, resp, history = self._fit_single(X, rng)
            if best is None or history[-1] > best[0]:
                best = (float(history[-1]), weights, class_probs, resp, history)
        assert best is not None
        _, weights, class_probs, resp, history = best
        self.weights_ = weights
        self.class_probs_ = class_probs
        self.responsibilities_ = resp
        self.labels_ = resp.argmax(axis=1)
        self.loglik_history_ = history
        self.log_likelihood_ = float(history[-1])
        return self
