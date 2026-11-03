"""Numerically stable log-sum-exp and the monotone-log-likelihood invariant.

The EM theory guarantees the (incomplete-data) log-likelihood never decreases
across iterations. We track it and assert it — a decrease is a bug, not noise.
"""

from __future__ import annotations

import numpy as np


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable ``log(sum(exp(a)))`` along ``axis`` (0-d array when ``axis`` is None)."""
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    summed = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    result = np.log(summed) + a_max
    return np.squeeze(result, axis=axis)


def is_monotone_nondecreasing(history: np.ndarray, atol: float = 1e-8) -> bool:
    """True when ``history`` never drops by more than ``atol`` between steps."""
    h = np.asarray(history, dtype=float)
    if h.size < 2:
        return True
    return bool(np.all(np.diff(h) >= -atol))


def assert_monotone(history: np.ndarray, atol: float = 1e-8) -> None:
    """Raise AssertionError at the first step where the log-likelihood drops."""
    h = np.asarray(history, dtype=float)
    diffs = np.diff(h)
    bad = np.where(diffs < -atol)[0]
    if bad.size:
        raise AssertionError(f"log-likelihood decreased at step(s) {bad.tolist()}")
