"""K-Means as hard EM.

K-Means is exactly the Gaussian-mixture loop with the soft responsibilities
forced to all-or-nothing (argmax), spherical equal covariance, and equal fixed
weights. ``KMeansEM`` is the standalone Lloyd implementation; ``kmeans_as_hard_em``
drives the shared GMM scaffold in that reduced regime so the equivalence is
literal in the code, not just in prose.
"""

from __future__ import annotations

import numpy as np

from ..diagnostics.init_strategies import kmeans_plus_plus
from .gmm import GaussianMixtureEM


class KMeansEM:
    """Lloyd's K-Means, framed as the hard-assignment special case of EM."""

    def __init__(
        self,
        n_clusters: int,
        *,
        max_iter: int = 300,
        tol: float = 1e-8,
        n_init: int = 10,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def _fit_single(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float]:
        centers = kmeans_plus_plus(X, self.n_clusters, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.max_iter):
            dist_sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # E-step (hard)
            labels = dist_sq.argmin(axis=1)
            new_centers = np.stack(
                [
                    X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                    for j in range(self.n_clusters)
                ]
            )  # M-step
            shift = float(((new_centers - centers) ** 2).sum())
            centers = new_centers
            if shift <= self.tol:
                break
        dist_sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = float(dist_sq[np.arange(X.shape[0]), labels].sum())
        return labels, centers, inertia

    def fit(self, X: np.ndarray) -> KMeansEM:
        """Fit via ``n_init`` restarts, keeping the lowest-inertia solution."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for _ in range(self.n_init):
            labels, centers, inertia = self._fit_single(X, rng)
            if best is None or inertia < best[0]:
                best = (inertia, labels, centers)
        assert best is not None
        inertia, labels, centers = best
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.inertia_ = inertia
        return self


def kmeans_as_hard_em(
    X: np.ndarray, n_clusters: int, *, max_iter: int = 300, random_state: int = 0
) -> np.ndarray:
    """Run the GMM loop in hard / spherical / tied / equal-weight mode; return labels.

    With equal fixed weights and a tied spherical covariance, the EM argmax
    reduces to nearest-centroid assignment — i.e. K-Means, assignment for
    assignment.
    """
    gmm = GaussianMixtureEM(
        n_clusters,
        covariance_type="spherical",
        tied=True,
        hard=True,
        update_weights=False,
        init="kmeans++",
        n_init=1,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(X)
    return gmm.labels_
