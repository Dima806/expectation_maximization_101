"""Two-coins EM recovers the planted biases."""

import numpy as np

from src.datasets.mixtures import make_two_coins
from src.em.two_coins import TwoCoinsEM


def test_recovers_planted_biases():
    heads, _, truth = make_two_coins(500, 20, [0.8, 0.35], seed=1)
    model = TwoCoinsEM(2, n_init=5, random_state=0).fit(heads, 20)
    recovered = np.sort(model.biases_)
    planted = np.sort(truth.biases)
    assert np.allclose(recovered, planted, atol=0.05)
