"""Latent class analysis recovers the planted hidden classes."""

import numpy as np

from src.datasets.mixtures import make_categorical_mixture
from src.em.latent_class import LatentClassEM
from src.evaluation.recovery import clustering_accuracy


def test_recovers_latent_classes():
    profile_a = np.tile([0.8, 0.1, 0.1], (8, 1))
    profile_b = np.tile([0.1, 0.1, 0.8], (8, 1))
    class_probs = np.stack([profile_a, profile_b])  # (2 classes, 8 features, 3 categories)
    X, labels, _ = make_categorical_mixture(800, [0.5, 0.5], class_probs, seed=3)
    lca = LatentClassEM(2, 3, n_init=5, random_state=0).fit(X)
    assert clustering_accuracy(labels, lca.labels_) > 0.9
