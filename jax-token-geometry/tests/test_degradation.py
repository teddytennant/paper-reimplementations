"""The failure direction Theorem 1 predicts.

A test that only passes for a well-fitted model says less than one that also
confirms the geometry degrades the way the error terms say it should. Theorem 1
has two of them: sqrt(epsilon_p) and 1/sqrt(p(v)). Both are checked here, the
first along a training path and the second by making one token type rare.
"""

import numpy as np

from tokengeom import downstream as D
from tokengeom import geometry as G
from tokengeom import train as T
from tokengeom.generative import SemanticModel, SimulationConfig


def model_hellinger(pair):
    e = pair.embeddings()
    d = len(e)
    out = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            out[i, j] = out[j, i] = 1.0 - np.exp(-pair.jensen_gap(e[i], e[j]))
    return out


def test_the_path_spans_a_wide_range_of_prediction_error(fit_path):
    errors = [pair.epsilon_p for pair in fit_path]
    assert max(errors) > 0.2
    assert min(errors) < 1e-3


def test_geometry_error_shrinks_with_prediction_error(fit_path, sim):
    """max_pairs |sqrt(Gamma_h) - H(model)| tracks epsilon_p along the path.

    Theorem 1 bounds that deviation by sqrt(epsilon_p / min p(v)) pairwise. The
    bound holds at every point on the path, and the deviation itself moves with
    epsilon_p rather than staying flat.
    """
    gamma = np.sqrt(np.asarray(sim.gamma_h_tokens))
    p_min = np.minimum.outer(sim.token_marginal, sim.token_marginal)
    errors, deviations = [], []
    for pair in fit_path:
        deviation = np.abs(np.sqrt(model_hellinger(pair)) - gamma)
        assert np.all(deviation <= np.sqrt(pair.epsilon_p / p_min) + 1e-12)
        errors.append(pair.epsilon_p)
        deviations.append(deviation.max())
    assert G.spearman(errors, deviations) > 0.9
    assert deviations[int(np.argmax(errors))] > 20 * deviations[int(np.argmin(errors))]


def test_equivalence_collapse_improves_with_prediction_error(fit_path, sim):
    """D_s of Section 5 falls as epsilon_p falls, panel (middle) of Figure 2."""
    errors = [pair.epsilon_p for pair in fit_path]
    ratios = [pair.synonym_distance_ratio(sim) for pair in fit_path]
    assert G.spearman(errors, ratios) > 0.9
    assert ratios[int(np.argmax(errors))] > 100 * ratios[int(np.argmin(errors))]


def test_community_recovery_degrades_with_prediction_error(fit_path, sim):
    """Figure 5 left: clustering works when the geometry is recovered.

    The misclustering rate falls monotonically along the training path, from
    half the vocabulary misassigned to exact recovery.
    """
    n_clusters = sim.config.n_communities
    errors, rates = [], []
    for pair in fit_path:
        whitened = G.whiten_coordinates(pair)
        labels, _, _, _, _ = D.spectral_clustering(
            np.asarray(whitened.embeddings()), n_clusters
        )
        errors.append(pair.epsilon_p)
        rates.append(D.misclustering_rate(labels, sim.token_community, n_clusters))
    assert G.spearman(errors, rates) > 0.9
    assert rates[int(np.argmax(errors))] >= 0.4
    assert rates[int(np.argmin(errors))] == 0.0


def rare_variant_model(rare_mass, beta):
    """The small simulation with one deliberately rare exchangeable variant."""
    weights = ((1.0 - rare_mass) / 2, (1.0 - rare_mass) / 2, rare_mass)
    config = SimulationConfig(
        n_communities=4,
        groups_per_community=2,
        n_variants=3,
        latent_dim=8,
        n_geometric=3,
        seed=0,
        variant_weights=weights,
        beta=beta,
    )
    return SemanticModel(config)


def within_group_distances(pair, sim):
    """Mean squared embedding distance for frequent-frequent and
    frequent-rare pairs, both of which have Gamma_h exactly zero."""
    d2 = pair.squared_embedding_distances()
    frequent, rare = [], []
    for a in range(sim.config.n_groups):
        i = np.where(sim.token_group == a)[0]
        frequent.append(d2[i[0], i[1]])
        rare.append(0.5 * (d2[i[0], i[2]] + d2[i[1], i[2]]))
    return float(np.mean(frequent)), float(np.mean(rare))


def test_collapse_degrades_for_rare_token_types():
    """Theorem 1 is only informative when both types are moderately frequent.

    Two token types of the same semantic group have Gamma_h = 0 whatever their
    frequencies, so any gap between their embeddings is the error term at work.
    Making one variant rare separates it from its synonyms by orders of
    magnitude more than the two frequent variants are separated from each other.
    """
    reference = SemanticModel(
        SimulationConfig(
            n_communities=4, groups_per_community=2, n_variants=3,
            latent_dim=8, n_geometric=3, seed=0,
        )
    )
    sim = rare_variant_model(0.02, reference.beta)
    enum = T.enumerate_semantic_contexts(sim)
    params, _ = T.fit_unrestricted(sim, enum, width=8, n_steps=150)
    pair = G.FittedPair.from_unrestricted(params, enum)
    frequent, rare = within_group_distances(pair, sim)
    assert rare > 50 * frequent


def test_collapse_scales_with_the_rare_frequency():
    """The 1/p(v) direction of the bound, as a rate.

    Holding the fit budget fixed and shrinking the rare variant's mass by 40x,
    the squared distance between the rare type and its synonyms grows
    monotonically. The theorem's constant is not explicit at this scale, so the
    claim under test is the direction and monotonicity in p(v), not a constant.
    """
    reference = SemanticModel(
        SimulationConfig(
            n_communities=4, groups_per_community=2, n_variants=3,
            latent_dim=8, n_geometric=3, seed=0,
        )
    )
    masses = (0.2, 0.1, 0.05, 0.02, 0.005)
    gaps, errors = [], []
    for mass in masses:
        sim = rare_variant_model(mass, reference.beta)
        enum = T.enumerate_semantic_contexts(sim)
        params, _ = T.fit_unrestricted(sim, enum, width=8, n_steps=120)
        pair = G.FittedPair.from_unrestricted(params, enum)
        _, rare = within_group_distances(pair, sim)
        gaps.append(rare)
        errors.append(pair.epsilon_p)
    assert np.all(np.diff(gaps) > 0)
    assert gaps[-1] > 20 * gaps[0]
    # the prediction error stays in the same decade, so the growth is the
    # frequency term and not a worse fit
    assert max(errors) / min(errors) < 10
