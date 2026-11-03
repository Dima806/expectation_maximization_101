"""The hard-EM path reproduces K-Means assignments, assignment for assignment."""

from src.datasets.mixtures import make_blobs
from src.em.kmeans import KMeansEM, kmeans_as_hard_em
from src.evaluation.recovery import same_partition


def test_hard_em_matches_kmeans():
    X, _, _ = make_blobs(600, [[-6, 0], [6, 0], [0, 7]], cluster_std=1.0, seed=7)
    km = KMeansEM(3, n_init=1, random_state=0).fit(X)
    hard_labels = kmeans_as_hard_em(X, 3, random_state=0)
    assert same_partition(km.labels_, hard_labels)
