"""Validate the from-scratch GMM against sklearn's GaussianMixture.

sklearn appears here (and only here, plus tests) as the reference oracle — never
inside the from-scratch EM implementations.
"""

from __future__ import annotations

from sklearn.mixture import GaussianMixture

from ..em.gmm import GaussianMixtureEM


def compare_gmm(
    X,
    n_components: int,
    *,
    random_state: int = 0,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_init: int = 10,
) -> dict:
    """Fit both implementations and compare average per-sample log-likelihood."""
    ours = GaussianMixtureEM(
        n_components, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state
    ).fit(X)
    sk = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    ).fit(X)
    ours_score = ours.score(X)
    sklearn_score = float(sk.score(X))
    return {
        "ours_avg_loglik": ours_score,
        "sklearn_avg_loglik": sklearn_score,
        "abs_diff": abs(ours_score - sklearn_score),
    }
