"""The canonical discrete EM example: two biased coins (Do & Batzoglou, 2008).

You are handed flip-count series from coins you cannot tell apart. If you knew
which coin produced each series you could estimate the biases; if you knew the
biases you could guess the coins. You know neither — so EM bootstraps:

  E-step: responsibility that each series came from each coin (binomial kernel).
  M-step: re-estimate each coin's bias, weighting every series by that responsibility.

Small enough to trace by hand, with no distributional machinery in the way.
"""

from __future__ import annotations

import numpy as np

from ..diagnostics.likelihood import logsumexp


class TwoCoinsEM:
    """EM for a mixture of ``n_coins`` Bernoulli coins observed as flip counts."""

    def __init__(
        self,
        n_coins: int = 2,
        *,
        max_iter: int = 200,
        tol: float = 1e-9,
        n_init: int = 5,
        random_state: int = 0,
    ) -> None:
        self.n_coins = n_coins
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def _responsibilities(
        self, heads: np.ndarray, flips: np.ndarray, biases: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, float]:
        theta = np.clip(biases, 1e-9, 1 - 1e-9)
        log_lik = (
            heads[:, None] * np.log(theta)[None, :]
            + (flips - heads)[:, None] * np.log(1 - theta)[None, :]
        )
        weighted = log_lik + np.log(np.clip(weights, 1e-300, None))
        log_norm = logsumexp(weighted, axis=1)
        resp = np.exp(weighted - log_norm[:, None])
        return resp, float(log_norm.sum())

    def _fit_single(
        self, heads: np.ndarray, flips: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        biases = rng.uniform(0.2, 0.8, self.n_coins)
        weights = np.full(self.n_coins, 1.0 / self.n_coins)
        history: list[float] = []
        for it in range(self.max_iter):
            resp, ll = self._responsibilities(heads, flips, biases, weights)
            history.append(ll)
            nk = resp.sum(axis=0) + 1e-12
            weights = nk / len(heads)
            biases = (resp * heads[:, None]).sum(axis=0) / (resp * flips[:, None]).sum(axis=0)
            if it > 0 and ll - history[-2] < self.tol:
                break
        resp, ll = self._responsibilities(heads, flips, biases, weights)
        history.append(ll)
        return biases, weights, resp, np.asarray(history)

    def fit(self, head_counts: np.ndarray, n_flips: np.ndarray | int) -> TwoCoinsEM:
        """Fit to ``head_counts`` (heads per series) given ``n_flips`` per series."""
        heads = np.asarray(head_counts, dtype=float)
        flips = np.broadcast_to(np.asarray(n_flips, dtype=float), heads.shape).astype(float)
        rng = np.random.default_rng(self.random_state)
        best: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for _ in range(self.n_init):
            biases, weights, resp, history = self._fit_single(heads, flips, rng)
            if best is None or history[-1] > best[0]:
                best = (float(history[-1]), biases, weights, resp, history)
        assert best is not None
        _, biases, weights, resp, history = best
        self.biases_ = biases
        self.weights_ = weights
        self.responsibilities_ = resp
        self.loglik_history_ = history
        self.log_likelihood_ = float(history[-1])
        return self
