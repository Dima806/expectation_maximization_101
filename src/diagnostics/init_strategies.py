"""Initialization strategies for EM: random seeding and k-means++.

Bad initialization is the single biggest reason EM lands in a local optimum,
so these are first-class diagnostics, not an afterthought.
"""

from __future__ import annotations

import numpy as np


def random_init(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n_clusters`` distinct data points as initial centers."""
    X = np.asarray(X, dtype=float)
    idx = rng.choice(X.shape[0], size=n_clusters, replace=False)
    return X[idx].copy()


def kmeans_plus_plus(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread initial centers by squared-distance sampling."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    first = int(rng.integers(n))
    centers = [X[first]]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for _ in range(1, n_clusters):
        total = closest_sq.sum()
        probs = closest_sq / total if total > 0 else np.full(n, 1.0 / n)
        idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])
        dist_sq = np.sum((X - X[idx]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
    return np.asarray(centers, dtype=float)
