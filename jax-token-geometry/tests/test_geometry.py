"""Theorem 1: token embeddings recover the Hellinger geometry of contexts.

The chain the paper builds in Section 2.3 is

    H^2(model-implied context distributions) = 1 - exp(-J_A(E_v1, E_v2)),
    b^2 ||E_v1 - E_v2||^2 <= 8 J_A <= B^2 ||E_v1 - E_v2||^2,
    sqrt(Gamma_h) within sqrt(epsilon_p / min(p(v1), p(v2))) of that,

and each link is checked here on the fitted pair. The curvature constants
b_Theta and B_Theta are measured on the fitted representation rather than
assumed, so no constant in these tests is invented.
"""

import numpy as np
import pytest

from tokengeom import geometry as G
from tokengeom.hellinger import pairwise_squared_hellinger


def test_fit_reaches_a_small_population_error(fitted):
    """The regime Theorem 1 is stated in: epsilon_p well below every p(v)."""
    assert fitted.epsilon_p < 1e-3


def test_exponential_family_identity(fitted):
    """H^2(p_hat_{E,v1}, p_hat_{E,v2}) = 1 - exp(-J_A(E_v1, E_v2)) exactly.

    This is the identity of Section 2.3, and it holds to machine precision
    because both sides are computed from the same enumerated population.
    """
    h2 = np.asarray(pairwise_squared_hellinger(fitted.model_context_distributions()))
    e = fitted.embeddings()
    d = h2.shape[0]
    worst = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            want = 1.0 - np.exp(-fitted.jensen_gap(e[i], e[j]))
            worst = max(worst, abs(h2[i, j] - want))
    assert worst < 1e-12


def test_fitted_marginal_matches_the_normalizer(fitted):
    """log p_hat(v) = b_v + A(E_v).

    The head bias factors out of both p_hat_E(v|Theta) and the induced marginal,
    which is why it drops out of p_hat_{E,v}(X|v) and leaves E as the natural
    parameter.
    """
    got = np.asarray(fitted.log_normalizer(fitted.embeddings()) + fitted.bias)
    assert np.abs(np.exp(got) - np.asarray(fitted.token_marginal)).max() < 1e-12


def test_curvature_sandwich(fitted, curvature):
    """b^2 ||dE||^2 / 8 <= J_A <= B^2 ||dE||^2 / 8 for every pair.

    Assumption 1.3 in the form Theorem 1 actually uses it.
    """
    lo, hi = curvature
    assert lo > 0.0
    e = fitted.embeddings()
    d2 = fitted.squared_embedding_distances()
    for i in range(d2.shape[0]):
        for j in range(i + 1, d2.shape[0]):
            gap = fitted.jensen_gap(e[i], e[j])
            assert gap >= lo / 8.0 * d2[i, j] - 1e-12
            assert gap <= hi / 8.0 * d2[i, j] + 1e-12


def test_theorem1_inequalities(sim, fitted, curvature):
    """Both bounds of Theorem 1, over every pair of token types.

    Nothing here is fitted to make the inequality pass: b_Theta, B_Theta,
    epsilon_p, p(v) and Gamma_h are all measured or exact.
    """
    lo, hi = curvature
    d2 = fitted.squared_embedding_distances()
    p_min = np.minimum.outer(sim.token_marginal, sim.token_marginal)
    root_gamma = np.sqrt(np.asarray(sim.gamma_h_tokens))
    upper = G.theorem1_upper(d2, hi, fitted.epsilon_p, p_min)
    lower = G.theorem1_lower(d2, lo, fitted.epsilon_p, p_min)
    i, j = np.triu_indices(d2.shape[0], k=1)
    assert np.all(root_gamma[i, j] <= upper[i, j] + 1e-12)
    assert np.all(root_gamma[i, j] >= lower[i, j] - 1e-12)

    # The bounds carry information. On the pairs with Gamma_h = 0 the upper
    # bound is what forces the collapse, and it is far below the trivial value
    # of one; on the pairs from different groups the lower bound is strictly
    # positive, so the theorem forbids their embeddings from coinciding.
    same = sim.token_group[:, None] == sim.token_group[None, :]
    assert upper[same & ~np.eye(len(same), dtype=bool)].max() < 0.25
    across = ~same
    assert (lower[across] > 0).mean() > 0.8


def test_theorem1_sharp_form(sim, fitted):
    """The same statement with the Jensen gap left exact instead of sandwiched.

    Theorem 1 relaxes J_A through b_Theta and B_Theta. Keeping J_A gives the
    two-sided bound the proof really produces, which is tight on both sides.
    """
    e = fitted.embeddings()
    d = len(e)
    slack = np.sqrt(fitted.epsilon_p / np.minimum.outer(sim.token_marginal, sim.token_marginal))
    root_gamma = np.sqrt(np.asarray(sim.gamma_h_tokens))
    widths = []
    for i in range(d):
        for j in range(i + 1, d):
            model = np.sqrt(1.0 - np.exp(-fitted.jensen_gap(e[i], e[j])))
            assert root_gamma[i, j] <= model + slack[i, j] + 1e-12
            assert root_gamma[i, j] >= model - slack[i, j] - 1e-12
            widths.append(slack[i, j])
    assert max(widths) < 0.2


def test_equivalence_collapse(sim, fitted, curvature):
    """Gamma_h = 0 implies the embeddings coincide, up to the Theorem 1 slack.

    Section 2.4 states the exact equivalence at zero prediction error and an
    approximate one otherwise, with the degree controlled by epsilon_p and the
    token frequencies.
    """
    lo, _ = curvature
    d2 = fitted.squared_embedding_distances()
    same = sim.token_group[:, None] == sim.token_group[None, :]
    off_diagonal = same & ~np.eye(len(d2), dtype=bool)
    p_min = np.minimum.outer(sim.token_marginal, sim.token_marginal)
    certified = -8.0 / lo * np.log(1.0 - np.minimum(fitted.epsilon_p / p_min, 1 - 1e-15))
    assert np.all(d2[off_diagonal] <= certified[off_diagonal])
    # and in fact the collapse is far tighter than the certificate
    assert d2[off_diagonal].max() < 1e-3
    assert fitted.synonym_distance_ratio(sim) < 1e-4


def test_semantic_geometry_is_recovered(sim, fitted):
    """D_g against Gamma_h over the group pairs, the panel (a) of Figure 1.

    The paper reports a Spearman correlation of 0.79 over its 120 group pairs.
    The bar here is lower because the curvature of A is far from constant, with
    B_Theta^2 / b_Theta^2 near a thousand, and Theorem 1 only ties the Euclidean
    distance to Gamma_h through that sandwich. The sharp version of the same
    claim, on the model-implied Hellinger distance, is the next test.
    """
    d_g = fitted.group_embedding_distances(sim)
    rho = G.spearman(G.upper_triangle(d_g), G.upper_triangle(sim.gamma_h_groups))
    assert rho > 0.6


def test_model_implied_geometry_tracks_gamma_h(fitted, sim):
    """1 - exp(-J_A) against Gamma_h, pair by pair.

    This is the quantity Theorem 1 actually controls, before the curvature
    sandwich turns it into a statement about Euclidean distance.
    """
    e = fitted.embeddings()
    d = len(e)
    model = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            model[i, j] = model[j, i] = 1.0 - np.exp(-fitted.jensen_gap(e[i], e[j]))
    gamma = np.asarray(sim.gamma_h_tokens)
    assert G.spearman(G.upper_triangle(model), G.upper_triangle(gamma)) > 0.98
    deviation = np.abs(np.sqrt(model) - np.sqrt(gamma)).max()
    assert deviation < np.sqrt(fitted.epsilon_p / sim.token_marginal.min())
    assert deviation < 0.05


def test_whitening_keeps_the_loss_and_sharpens_the_geometry(sim, fitted):
    """Section 3.1, made concrete: predictively identical, geometrically better.

    All representatives of the equivalence class have the same prediction loss,
    but Theorem 1 reads Gamma_h off Euclidean distances, so the coordinates
    matter. Rescaling by the inverse square root of the curvature at the
    embedding centroid changes nothing the objective can see and raises the
    correlation between D_g and Gamma_h.
    """
    whitened = G.whiten_coordinates(fitted)
    assert whitened.epsilon_p == pytest.approx(fitted.epsilon_p, rel=1e-10)
    gamma = G.upper_triangle(sim.gamma_h_groups)
    before = G.spearman(G.upper_triangle(fitted.group_embedding_distances(sim)), gamma)
    after = G.spearman(G.upper_triangle(whitened.group_embedding_distances(sim)), gamma)
    assert after > before
    assert after > 0.8


def test_prediction_loss_is_invariant_to_coordinate_change(sim, fitted):
    """Section 3.1: (A Theta, A^{-T} E) leaves every kernel score alone.

    So the token-prediction objective identifies an equivalence class of dual
    representations, not a coordinate system.
    """
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.standard_normal((fitted.theta.shape[1],) * 2)
        moved = G.coordinate_change(fitted, a)
        assert moved.epsilon_p == pytest.approx(fitted.epsilon_p, rel=1e-10)
        assert np.abs(
            np.asarray(moved.log_predictions) - np.asarray(fitted.log_predictions)
        ).max() < 1e-10
