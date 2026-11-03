"""The log-likelihood never decreases across iterations (the EM invariant)."""

from src.datasets.mixtures import make_blobs, make_missing_gaussian, make_two_coins
from src.diagnostics.likelihood import is_monotone_nondecreasing
from src.em.gmm import GaussianMixtureEM
from src.em.missing_data import GaussianMissingEM
from src.em.two_coins import TwoCoinsEM


def test_gmm_monotone():
    X, _, _ = make_blobs(500, [[-3, 0], [3, 0]], seed=5)
    gmm = GaussianMixtureEM(2, n_init=1, random_state=0).fit(X)
    assert is_monotone_nondecreasing(gmm.loglik_history_, atol=1e-4)


def test_two_coins_monotone():
    heads, _, _ = make_two_coins(300, 15, [0.7, 0.3], seed=1)
    model = TwoCoinsEM(2, n_init=1, random_state=0).fit(heads, 15)
    assert is_monotone_nondecreasing(model.loglik_history_, atol=1e-4)


def test_missing_data_monotone():
    _, X_missing, _ = make_missing_gaussian(
        500, [0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], missing_frac=0.2, seed=4
    )
    model = GaussianMissingEM().fit(X_missing)
    assert is_monotone_nondecreasing(model.loglik_history_, atol=1e-4)
