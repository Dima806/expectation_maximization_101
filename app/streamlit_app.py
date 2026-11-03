"""Streamlit demos for EM: animator, soft-to-hard, restart roulette, missing data.

Run with: make run  (uv run streamlit run app/streamlit_app.py)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.datasets.mixtures import make_blobs, make_missing_gaussian
from src.em.gmm import GaussianMixtureEM
from src.em.kmeans import KMeansEM
from src.em.missing_data import GaussianMissingEM
from src.evaluation.local_optima import restart_spread
from src.visualisation import plot_clusters_2d, plot_loglik

st.set_page_config(page_title="Expectation-Maximization 101", layout="wide")
st.title("Expectation-Maximization 101")

demo = st.sidebar.radio(
    "Demo",
    ["EM animator", "Soft-to-hard slider", "Restart roulette", "Missing-data demo"],
)
seed = st.sidebar.number_input("seed", value=0, step=1)


def _two_cluster_data(sep: float, n: int, rng_seed: int) -> np.ndarray:
    centers = np.array([[-sep / 2, 0.0], [sep / 2, 0.0]])
    X, _, _ = make_blobs(n, centers, cluster_std=1.0, seed=rng_seed)
    return X


if demo == "EM animator":
    st.header("Watch the mixture separate and the log-likelihood climb")
    sep = st.slider("component separation", 1.0, 8.0, 3.0, 0.5)
    n_iter = st.slider("iterations", 1, 40, 15)
    X = _two_cluster_data(sep, 600, int(seed))
    gmm = GaussianMixtureEM(2, n_init=1, max_iter=n_iter, random_state=int(seed)).fit(X)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        plot_clusters_2d(X, gmm.labels_, centers=gmm.means_, ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        plot_loglik(gmm.loglik_history_, ax=ax)
        st.pyplot(fig)

elif demo == "Soft-to-hard slider":
    st.header("Soft responsibilities → hard assignments = K-Means")
    hardness = st.slider("hardness (0 = soft GMM, 1 = hard / K-Means)", 0.0, 1.0, 0.0, 1.0)
    X = _two_cluster_data(4.0, 600, int(seed))
    is_hard = hardness >= 0.5
    gmm = GaussianMixtureEM(2, hard=is_hard, n_init=1, random_state=int(seed)).fit(X)
    km = KMeansEM(2, n_init=1, random_state=int(seed)).fit(X)
    fig, ax = plt.subplots()
    plot_clusters_2d(X, gmm.labels_, centers=gmm.means_, ax=ax)
    ax.set_title("hard EM (= K-Means)" if is_hard else "soft GMM")
    st.pyplot(fig)
    st.caption(f"K-Means inertia at this data: {km.inertia_:.1f}")

elif demo == "Restart roulette":
    st.header("Many random starts land in different basins")
    n_restarts = st.slider("random restarts", 5, 60, 30)
    X = _two_cluster_data(1.5, 600, int(seed))
    spread = restart_spread(X, 3, n_restarts=n_restarts)
    fig, ax = plt.subplots()
    ax.hist(spread["log_likelihoods"], bins=20, color="steelblue")
    ax.set_xlabel("final log-likelihood")
    ax.set_ylabel("count")
    st.pyplot(fig)
    st.metric("fraction stuck below best", f"{spread['suboptimal_fraction']:.0%}")

else:
    st.header("EM imputation restores the correlation mean-filling destroys")
    corr = st.slider("true correlation", 0.0, 0.95, 0.8, 0.05)
    frac = st.slider("missing fraction", 0.05, 0.6, 0.25, 0.05)
    cov = np.array([[1.0, corr], [corr, 1.0]])
    _, X_missing, _ = make_missing_gaussian(
        2000, [0.0, 0.0], cov, missing_frac=frac, seed=int(seed)
    )
    em = GaussianMissingEM(random_state=int(seed)).fit(X_missing)
    observed = ~np.isnan(X_missing)
    col_mean = np.nanmean(X_missing, axis=0)
    filled = np.where(observed, X_missing, col_mean)
    mf_cov = np.cov(filled.T)
    corr_em = em.covariance_[0, 1] / np.sqrt(em.covariance_[0, 0] * em.covariance_[1, 1])
    corr_mf = mf_cov[0, 1] / np.sqrt(mf_cov[0, 0] * mf_cov[1, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("true corr", f"{corr:.2f}")
    c2.metric("EM-imputed corr", f"{corr_em:.2f}")
    c3.metric("mean-fill corr", f"{corr_mf:.2f}")
