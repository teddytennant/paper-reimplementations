"""Section 4.3 and Theorem 6: a linear probe on the contextual representation.

The task is the one Figure 6 uses, predicting the semantic community of the
target token from the representation of its context. The excess risk is measured
against the exact Bayes rule eta_k(X_{j,c}) = P(Y = k | X_{j,c}), which the
enumeration gives in closed form.

Theorem 6's bound is

    R <= C (estimation/n + 2 epsilon_p/rho_0 + D^4 B_Theta^4/16 + C_c3 delta_s)^{(beta+1)/(beta+2)}

with C_{c,1}, ..., C_{c,4} and n_0 defined inside the proof and not given
explicitly. Those constants are not invented here: what is tested is the shape
of the statement, that excess risk falls with n, that it falls as epsilon_p
falls, and that the contextual representation beats a representation carrying no
contextual information. The one term that is explicit, the certified embedding
diameter D of a preferred set, is checked as an inequality.
"""

import numpy as np
import pytest

from tokengeom import downstream as D
from tokengeom import geometry as G
from tokengeom import model as M

SAMPLE_SIZES = (50, 200, 1000)
LABEL_SEEDS = (11, 12, 13)


def average_excess_risk(features, posteriors, context_dist, n, seeds, n_classes):
    risks = []
    for seed in seeds:
        idx, labels = D.sample_labeled(posteriors, context_dist, n, seed)
        probe = D.fit_probe(np.asarray(features)[idx], labels, n_classes)
        risks.append(D.probe_excess_risk(probe, features, posteriors, context_dist))
    return float(np.mean(risks))


def test_posteriors_and_bayes_rule(sim, token_enum):
    eta = D.class_posteriors(sim, token_enum)
    assert np.allclose(eta.sum(axis=1), 1.0, atol=1e-12)
    risk = D.bayes_risk(eta, token_enum.context_prob)
    assert 0.0 < risk < 1.0 - 1.0 / sim.config.n_communities
    # the label is genuinely uncertain given the context, so the task is not
    # solved by memorizing the preceding token
    assert risk > 0.2


def test_excess_risk_is_nonnegative_and_falls_with_sample_size(sim, encoder, token_enum):
    """The n term of Theorem 6, at the three sample sizes of Figure 6."""
    _, _, pair = encoder
    eta = D.class_posteriors(sim, token_enum)
    risks = [
        average_excess_risk(
            pair.theta, eta, token_enum.context_prob, n, LABEL_SEEDS,
            sim.config.n_communities,
        )
        for n in SAMPLE_SIZES
    ]
    assert all(risk >= -1e-12 for risk in risks)
    assert np.all(np.diff(risks) < 0)
    assert risks[-1] < 0.5 * risks[0]


def test_contextual_representation_beats_the_preceding_token(sim, encoder, token_enum):
    """Figure 6's baseline: a one-hot of the token just before the target.

    It carries no contextual information, and the probe on it cannot reach the
    Bayes rule however many labels it gets.
    """
    _, _, pair = encoder
    eta = D.class_posteriors(sim, token_enum)
    baseline = D.one_hot_previous_token(token_enum, sim.config.vocab_size)
    n = SAMPLE_SIZES[-1]
    learned = average_excess_risk(
        pair.theta, eta, token_enum.context_prob, n, LABEL_SEEDS, sim.config.n_communities
    )
    plain = average_excess_risk(
        baseline, eta, token_enum.context_prob, n, LABEL_SEEDS, sim.config.n_communities
    )
    assert learned < plain


def test_decoder_optimal_representation_is_the_best_feature(sim, encoder, ideal, token_enum):
    """Figure 6: Z_j(X) generally gives the most accurate classifier."""
    _, _, pair = encoder
    eta = D.class_posteriors(sim, token_enum)
    n = SAMPLE_SIZES[-1]
    optimal = average_excess_risk(
        ideal, eta, token_enum.context_prob, n, LABEL_SEEDS, sim.config.n_communities
    )
    learned = average_excess_risk(
        pair.theta, eta, token_enum.context_prob, n, LABEL_SEEDS, sim.config.n_communities
    )
    assert optimal <= learned + 1e-9


def test_probe_improves_as_prediction_error_falls(sim, fit_path, semantic_enum):
    """The 2 epsilon_p / rho_0 term of Theorem 6, along the training path."""
    eta = D.class_posteriors(sim, semantic_enum)
    errors = [pair.epsilon_p for pair in fit_path]
    worst = fit_path[int(np.argmax(errors))]
    best = fit_path[int(np.argmin(errors))]
    risks = [
        average_excess_risk(
            pair.theta, eta, semantic_enum.context_prob, SAMPLE_SIZES[-1], LABEL_SEEDS,
            sim.config.n_communities,
        )
        for pair in (worst, best)
    ]
    assert risks[1] < risks[0]


def test_probe_respects_the_centering_constraint(sim, encoder, token_enum):
    """sum_k W_k = 0 and sum_k b_k = 0, the parameterization of Section 4.3."""
    _, _, pair = encoder
    eta = D.class_posteriors(sim, token_enum)
    idx, labels = D.sample_labeled(eta, token_enum.context_prob, 200, seed=21)
    probe = D.fit_probe(np.asarray(pair.theta)[idx], labels, sim.config.n_communities)
    assert np.abs(np.asarray(probe["w"]).sum(axis=0)).max() < 1e-12
    assert abs(float(np.asarray(probe["b"]).sum())) < 1e-12


def test_theorem6_embedding_diameter(sim, fitted, curvature):
    """||Ehat_v1 - Ehat_v2||^2 <= D^2 inside a preferred set W_k.

    With the label equal to the community of the target token, W_k is community
    k for any threshold tau below one, and g_w is the largest Hellinger distance
    between the class-conditional context distributions inside it, which equals
    the largest within-community Gamma_h. Assumption 5.4 is checked before the
    bound is used.
    """
    tau = 0.5
    lo, _ = curvature
    gamma = np.asarray(sim.gamma_h_tokens)
    community = sim.token_community
    off = ~np.eye(len(community), dtype=bool)
    same = (community[:, None] == community[None, :]) & off
    g_w = float(np.sqrt(gamma[same].max()))
    p_min = float(sim.token_marginal.min())

    inner = np.sqrt(1.0 - tau + tau * g_w**2) + np.sqrt(fitted.epsilon_p / p_min)
    assert inner < 1.0  # Assumption 5.4
    diameter_sq = -8.0 / lo * np.log(1.0 - inner**2)
    assert np.isfinite(diameter_sq)

    d2 = fitted.squared_embedding_distances()
    assert d2[same].max() <= diameter_sq
    # the intermediate claim the bound rests on
    assert gamma[same].max() <= 1.0 - tau + tau * g_w**2 + 1e-12


def test_representation_norm_is_the_theorem6_constant(encoder):
    """B_Theta of Assumption 5.2 is exactly 4 by construction, not estimated."""
    _, _, pair = encoder
    norms = np.linalg.norm(np.asarray(pair.theta), axis=1)
    assert norms.max() == pytest.approx(M.REPRESENTATION_NORM, rel=1e-12)
    assert norms.min() == pytest.approx(M.REPRESENTATION_NORM, rel=1e-12)
