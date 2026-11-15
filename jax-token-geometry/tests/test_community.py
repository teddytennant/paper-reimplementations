"""Section 4.2 and Theorem 5: recovering token communities by clustering.

Theorem 5 bounds the misclustering rate by 64(2+kappa)^2 K d_max / d times
((d - d_min) beta_sigma / (alpha_sigma d_min))^2, where alpha_sigma lower bounds
the within-community affinities and beta_sigma upper bounds the between ones.

Two regimes are separated. On planted embeddings, where alpha and beta are the
realized affinity extremes, Assumption 4.4 holds and the bound is checked
directly and is not vacuous. On the fitted embeddings, alpha_sigma and beta_sigma
have to come through Theorem 1, and at the epsilon_p and curvature ratio reached
here they are far too weak to satisfy Assumption 4.4, so what is checked is the
affinity sandwich itself and the recovery the paper reports in Figure 5.
"""

import itertools

import numpy as np
import pytest

from tokengeom import downstream as D
from tokengeom import geometry as G


def planted_embeddings(rng, n_clusters, per_cluster, separation, spread, dim=None):
    dim = dim or n_clusters
    centers = np.eye(n_clusters, dim) * separation
    labels = np.repeat(np.arange(n_clusters), per_cluster)
    noise = rng.standard_normal((n_clusters * per_cluster, dim))
    noise *= spread / np.linalg.norm(noise, axis=1, keepdims=True)
    return centers[labels] + noise, labels


def test_kmeans_recovers_a_well_separated_partition():
    rng = np.random.default_rng(0)
    points, labels = planted_embeddings(rng, 4, 6, separation=10.0, spread=0.2)
    found, objective = D.kmeans(points, 4, seed=0, n_restarts=10)
    assert D.misclustering_rate(found, labels, 4) == 0.0
    assert objective == pytest.approx(
        sum(((points[labels == k] - points[labels == k].mean(0)) ** 2).sum() for k in range(4))
    )


def test_laplacian_matches_the_definition():
    """W symmetric with unit diagonal, L = D - W positive semidefinite with the
    constant vector in its kernel."""
    rng = np.random.default_rng(1)
    points, _ = planted_embeddings(rng, 3, 4, separation=3.0, spread=0.5)
    d2 = D.squared_distances(points)
    w, values, vectors = D.laplacian_eigen(d2, sigma=1.0)
    assert np.allclose(w, w.T, atol=1e-15)
    assert np.allclose(np.diag(w), 1.0, atol=1e-15)
    assert values.min() > -1e-12
    assert values[0] == pytest.approx(0.0, abs=1e-12)
    constant = np.ones(len(points)) / np.sqrt(len(points))
    assert abs(abs(vectors[:, 0] @ constant) - 1.0) < 1e-10


def test_misclustering_rate_is_permutation_invariant():
    truth = np.repeat(np.arange(4), 4)
    for perm in itertools.permutations(range(4)):
        assert D.misclustering_rate(np.asarray(perm)[truth], truth, 4) == 0.0
    wrong = truth.copy()
    wrong[0] = (wrong[0] + 1) % 4
    assert D.misclustering_rate(wrong, truth, 4) == pytest.approx(1 / 16)


def test_theorem5_bound_on_planted_embeddings():
    """Theorem 5 with alpha and beta measured, in a regime where it bites.

    alpha_sigma and beta_sigma are exactly what the paper says they are, the
    within-community affinity floor and the between-community ceiling, and
    Assumption 4.4's ratio condition is verified rather than assumed.
    """
    rng = np.random.default_rng(2)
    n_clusters, per_cluster = 4, 4
    points, labels = planted_embeddings(
        rng, n_clusters, per_cluster, separation=6.0, spread=0.1
    )
    sizes = [per_cluster] * n_clusters
    d2 = D.squared_distances(points)
    sigma = 1.0
    found, objective, _, _, w = D.spectral_clustering_from_distances(
        d2, n_clusters, sigma=sigma, n_restarts=40
    )
    same = (labels[:, None] == labels[None, :]) & ~np.eye(len(labels), dtype=bool)
    alpha = float(w[same].min())
    beta = float(w[~same & ~np.eye(len(labels), dtype=bool)].max())

    ratio, threshold = D.assumption4_ratio(alpha, beta, n_clusters, sizes)
    assert ratio > threshold  # Assumption 4.4 holds
    bound = D.theorem5_bound(alpha, beta, n_clusters, sizes)
    assert bound < 1.0  # and the bound says something
    assert D.misclustering_rate(found, labels, n_clusters) <= bound


def test_theorem5_bound_degrades_as_the_communities_merge():
    """The bound grows as alpha/beta falls, and the clustering follows it."""
    n_clusters, per_cluster = 4, 4
    sizes = [per_cluster] * n_clusters
    bounds, rates = [], []
    for separation in (6.0, 3.0, 1.5, 0.5, 0.15):
        points, labels = planted_embeddings(
            np.random.default_rng(3), n_clusters, per_cluster,
            separation=separation, spread=0.1,
        )
        d2 = D.squared_distances(points)
        found, _, _, _, w = D.spectral_clustering_from_distances(
            d2, n_clusters, sigma=1.0, n_restarts=40
        )
        off = ~np.eye(len(labels), dtype=bool)
        same = (labels[:, None] == labels[None, :]) & off
        alpha = float(w[same].min())
        beta = float(w[~same & off].max())
        bounds.append(D.theorem5_bound(alpha, beta, n_clusters, sizes))
        rates.append(D.misclustering_rate(found, labels, n_clusters))
    assert np.all(np.diff(bounds) > 0)
    assert bounds[0] < 1.0 < bounds[-1]
    assert rates[0] == 0.0
    assert rates[-1] > 0.0


def test_kmeans_approximation_constant(sim, fitted):
    """Assumption 4.3: the partition found is within (1 + kappa)^2 of optimal.

    kappa is measured against the best objective over many restarts rather than
    assumed to be zero.
    """
    embeddings = np.asarray(G.whiten_coordinates(fitted).embeddings())
    d2 = D.squared_distances(embeddings)
    sigma = D.select_sigma(d2, sim.config.n_communities)
    _, values, vectors = D.laplacian_eigen(d2, sigma)
    u = vectors[:, : sim.config.n_communities]
    _, cheap = D.kmeans(u, sim.config.n_communities, seed=0, n_restarts=5)
    _, best = D.kmeans(u, sim.config.n_communities, seed=1, n_restarts=200)
    assert cheap >= best - 1e-12
    kappa = np.sqrt(cheap / max(best, 1e-30)) - 1.0
    assert kappa < 0.1


def test_learned_embeddings_recover_the_planted_communities(sim, fitted):
    """Figure 5: the block structure of Gamma_h shows up in the learned geometry.

    The oracle reference is the same pipeline run on the exact Gamma_h.
    """
    n_clusters = sim.config.n_communities
    oracle_labels, _, _, _, _ = D.spectral_clustering_from_distances(
        D.gamma_h_distances(np.asarray(sim.gamma_h_tokens)), n_clusters
    )
    assert D.misclustering_rate(oracle_labels, sim.token_community, n_clusters) == 0.0

    embeddings = np.asarray(G.whiten_coordinates(fitted).embeddings())
    labels, _, _, _, _ = D.spectral_clustering(embeddings, n_clusters)
    assert D.misclustering_rate(labels, sim.token_community, n_clusters) == 0.0


def test_affinity_sandwich_from_theorem1(sim, fitted):
    """alpha_sigma <= within-community affinities, between <= beta_sigma.

    The step Section 4.2 takes between Theorem 1 and Theorem 5. It holds, but
    the ratio alpha_sigma / beta_sigma is orders of magnitude below what
    Assumption 4.4 asks for, because b_Theta^2 and B_Theta^2 differ by about a
    factor of a thousand on this fit. That is why Theorem 5's own bound is
    tested on planted embeddings instead.
    """
    n_clusters = sim.config.n_communities
    whitened = G.whiten_coordinates(fitted)
    lo, hi = whitened.curvature_bounds()
    embeddings = np.asarray(whitened.embeddings())
    d2 = D.squared_distances(embeddings)
    sigma = D.select_sigma(d2, n_clusters)
    w, _, _ = D.laplacian_eigen(d2, sigma)

    gamma = np.asarray(sim.gamma_h_tokens)
    community = sim.token_community
    off = ~np.eye(len(community), dtype=bool)
    same = (community[:, None] == community[None, :]) & off
    g_within = float(gamma[same].max())
    g_between = float(gamma[~same & off].min())

    alpha, beta = D.affinity_bounds(
        g_within, g_between, fitted.epsilon_p, float(sim.token_marginal.min()),
        sigma, lo, hi,
    )
    assert w[same].min() >= alpha
    assert w[~same & off].max() <= beta
    ratio, threshold = D.assumption4_ratio(alpha, beta, n_clusters,
                                           [len(community) // n_clusters] * n_clusters)
    assert ratio < threshold
