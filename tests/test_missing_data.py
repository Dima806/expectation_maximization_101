"""EM imputation recovers covariance structure better than single mean-filling."""

import numpy as np

from src.datasets.mixtures import make_missing_gaussian
from src.em.missing_data import GaussianMissingEM


def _corr(cov: np.ndarray) -> float:
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def test_em_beats_meanfill_on_covariance():
    true_corr = 0.8
    true_cov = np.array([[1.0, true_corr], [true_corr, 1.0]])
    _, X_missing, _ = make_missing_gaussian(2000, [0.0, 0.0], true_cov, missing_frac=0.25, seed=11)

    em = GaussianMissingEM(random_state=0).fit(X_missing)
    corr_em = _corr(em.covariance_)

    observed = ~np.isnan(X_missing)
    col_mean = np.nanmean(X_missing, axis=0)
    filled = np.where(observed, X_missing, col_mean)
    corr_meanfill = _corr(np.cov(filled.T))

    # mean-filling attenuates correlation toward zero; EM preserves it
    assert corr_meanfill < corr_em
    assert abs(corr_em - true_corr) < abs(corr_meanfill - true_corr)
