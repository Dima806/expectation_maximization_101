"""Planted mixtures with known truth.

EM's promise is recovering hidden structure, and recovery can only be verified
against planted truth — so every generator returns the parameters it used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianMixtureTruth:
    """Ground-truth parameters of a Gaussian mixture."""

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


@dataclass
class TwoCoinsTruth:
    """Ground-truth biases and mixing weights of the two-coins problem."""

    biases: np.ndarray
    weights: np.ndarray


@dataclass
class LatentClassTruth:
    """Ground-truth parameters of a categorical (latent-class) mixture."""

    weights: np.ndarray
    class_probs: np.ndarray  # shape (n_classes, n_features, n_categories)


@dataclass
class MissingDataTruth:
    """Ground-truth mean/covariance and the observed-entry mask."""

    mean: np.ndarray
    covariance: np.ndarray
    mask: np.ndarray  # True where observed


def make_gaussian_mixture(
    n_samples: int,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, GaussianMixtureTruth]:
    """Sample a D-dimensional Gaussian mixture; return X, labels, truth."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    k = len(weights)
    counts = rng.multinomial(n_samples, weights)
    chunks = []
    labels = []
    for j in range(k):
        chunks.append(rng.multivariate_normal(means[j], covariances[j], size=int(counts[j])))
        labels.append(np.full(int(counts[j]), j, dtype=int))
    X = np.vstack(chunks)
    label_arr = np.concatenate(labels)
    perm = rng.permutation(n_samples)
    truth = GaussianMixtureTruth(weights, means, covariances)
    return X[perm], label_arr[perm], truth


def make_gaussian_mixture_1d(
    n_samples: int,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, GaussianMixtureTruth]:
    """1-D convenience wrapper returning a flat X of shape (n_samples,)."""
    means_2d = np.asarray(means, dtype=float).reshape(-1, 1)
    covs = np.asarray([[[float(s) ** 2]] for s in stds], dtype=float)
    X, labels, truth = make_gaussian_mixture(n_samples, weights, means_2d, covs, seed=seed)
    return X[:, 0], labels, truth


def make_blobs(
    n_samples: int,
    centers: np.ndarray,
    *,
    cluster_std: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, GaussianMixtureTruth]:
    """Equal-weight spherical blobs — the clean case for K-Means / GMM demos."""
    centers = np.asarray(centers, dtype=float)
    k, d = centers.shape
    covs = np.stack([np.eye(d) * cluster_std**2 for _ in range(k)])
    weights = np.full(k, 1.0 / k)
    return make_gaussian_mixture(n_samples, weights, centers, covs, seed=seed)


def make_two_coins(
    n_experiments: int,
    n_flips: int,
    biases: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, TwoCoinsTruth]:
    """Flip-count series from indistinguishable biased coins; return heads, coin, truth."""
    rng = np.random.default_rng(seed)
    biases = np.asarray(biases, dtype=float)
    k = len(biases)
    weight_arr = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, dtype=float)
    weight_arr = weight_arr / weight_arr.sum()
    coin = rng.choice(k, size=n_experiments, p=weight_arr)
    heads = rng.binomial(n_flips, biases[coin])
    return heads.astype(int), coin, TwoCoinsTruth(biases, weight_arr)


def make_categorical_mixture(
    n_samples: int,
    weights: np.ndarray,
    class_probs: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, LatentClassTruth]:
    """Survey-style categorical data from a latent-class model; return X, labels, truth."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    class_probs = np.asarray(class_probs, dtype=float)  # (k, m, c)
    k, m, c = class_probs.shape
    labels = rng.choice(k, size=n_samples, p=weights)
    X = np.empty((n_samples, m), dtype=int)
    for i in range(n_samples):
        cls = int(labels[i])
        for j in range(m):
            X[i, j] = int(rng.choice(c, p=class_probs[cls, j]))
    return X, labels, LatentClassTruth(weights, class_probs)


def make_missing_gaussian(
    n_samples: int,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    missing_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, MissingDataTruth]:
    """Correlated Gaussian with MCAR holes; return X_full, X_missing (NaNs), truth."""
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    X_full = rng.multivariate_normal(mean, covariance, size=n_samples)
    mask = rng.random(X_full.shape) >= missing_frac  # True where observed
    for i in range(n_samples):  # never blank an entire row
        if not mask[i].any():
            mask[i, int(rng.integers(X_full.shape[1]))] = True
    X_missing = X_full.copy()
    X_missing[~mask] = np.nan
    return X_full, X_missing, MissingDataTruth(mean, covariance, mask)
