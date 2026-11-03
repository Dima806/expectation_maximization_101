"""Scoring recovered structure against planted truth.

Component labels are arbitrary, so parameters and labelings are compared only
after an optimal matching (Hungarian assignment).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..datasets.mixtures import GaussianMixtureTruth


def match_components(true_means: np.ndarray, est_means: np.ndarray) -> np.ndarray:
    """Return ``perm`` such that ``est_means[perm]`` best matches ``true_means``."""
    true_means = np.asarray(true_means, dtype=float)
    est_means = np.asarray(est_means, dtype=float)
    cost = np.linalg.norm(true_means[:, None, :] - est_means[None, :, :], axis=2)
    _, col = linear_sum_assignment(cost)
    return col


def parameter_recovery(
    truth: GaussianMixtureTruth,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> dict:
    """Per-component errors between recovered and planted GMM parameters."""
    perm = match_components(truth.means, means)
    mean_err = np.linalg.norm(truth.means - means[perm], axis=1)
    weight_err = np.abs(truth.weights - weights[perm])
    cov_err = np.array(
        [np.linalg.norm(truth.covariances[i] - covariances[perm][i]) for i in range(len(perm))]
    )
    return {
        "perm": perm,
        "mean_error": mean_err,
        "weight_error": weight_err,
        "cov_error": cov_err,
        "max_mean_error": float(mean_err.max()),
        "max_weight_error": float(weight_err.max()),
        "max_cov_error": float(cov_err.max()),
    }


def clustering_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Best label-permutation accuracy between two hard clusterings."""
    true = np.asarray(true_labels, dtype=int)
    pred = np.asarray(pred_labels, dtype=int)
    k = int(max(true.max(), pred.max())) + 1
    confusion = np.zeros((k, k), dtype=int)
    for t, p in zip(true, pred, strict=False):
        confusion[t, p] += 1
    row, col = linear_sum_assignment(-confusion)
    return float(confusion[row, col].sum()) / len(true)


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two labelings induce the same partition (labels may differ)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    a_to_b: dict[int, int] = {}
    b_to_a: dict[int, int] = {}
    for x, y in zip(a.tolist(), b.tolist(), strict=False):
        if a_to_b.setdefault(x, y) != y or b_to_a.setdefault(y, x) != x:
            return False
    return True
