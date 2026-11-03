"""From-scratch GMM matches sklearn and recovers planted parameters."""

import numpy as np

from src.datasets.mixtures import make_blobs, make_gaussian_mixture
from src.em.gmm import GaussianMixtureEM
from src.evaluation.recovery import parameter_recovery
from src.evaluation.vs_sklearn import compare_gmm


def test_matches_sklearn():
    X, _, _ = make_blobs(600, [[-4, 0], [4, 0], [0, 5]], cluster_std=1.0, seed=3)
    result = compare_gmm(X, 3, random_state=0, n_init=10)
    assert result["abs_diff"] < 0.05


def test_recovers_planted_params():
    means = np.array([[-5.0, 0.0], [5.0, 0.0]])
    covs = np.array([np.eye(2), np.eye(2) * 1.5])
    X, _, truth = make_gaussian_mixture(1000, [0.5, 0.5], means, covs, seed=2)
    gmm = GaussianMixtureEM(2, n_init=10, random_state=0).fit(X)
    rec = parameter_recovery(truth, gmm.weights_, gmm.means_, gmm.covariances_)
    assert rec["max_mean_error"] < 0.4
    assert rec["max_weight_error"] < 0.1
