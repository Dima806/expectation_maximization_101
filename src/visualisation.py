"""Shared plotting helpers for the notebooks and the Streamlit app."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_loglik(history: np.ndarray, *, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the EM log-likelihood against iteration (should climb monotonically)."""
    if ax is None:
        _, ax = plt.subplots()
    history = np.asarray(history, dtype=float)
    ax.plot(np.arange(history.size), history, marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title("EM log-likelihood (monotone)")
    return ax


def plot_gmm_1d(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    bins: int = 40,
) -> plt.Axes:
    """Histogram of 1-D data with the recovered mixture components overlaid."""
    if ax is None:
        _, ax = plt.subplots()
    X = np.asarray(X, dtype=float)
    ax.hist(X, bins=bins, density=True, alpha=0.4, color="gray")
    grid = np.linspace(X.min(), X.max(), 300)
    for w, m, s in zip(weights, means, stds, strict=False):
        pdf = w * np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        ax.plot(grid, pdf)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    return ax


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    centers: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter 2-D points colored by cluster, with optional centers marked."""
    if ax is None:
        _, ax = plt.subplots()
    X = np.asarray(X, dtype=float)
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=8, cmap="tab10")
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", c="black", s=120)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return ax
