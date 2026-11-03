"""BIC recovers the true component count on well-separated data."""

from src.datasets.mixtures import make_blobs
from src.diagnostics.model_selection import select_n_components


def test_bic_picks_true_k():
    X, _, _ = make_blobs(900, [[-8, 0], [8, 0], [0, 9]], cluster_std=1.0, seed=9)
    result = select_n_components(X, range(1, 6), n_init=5, random_state=0)
    assert result["best_bic_k"] == 3
