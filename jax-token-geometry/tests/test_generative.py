"""The ground-truth process of Appendix B.1, checked against its own definition.

Every later test reads population quantities off this object, so the exactness
claims of Section 5 have to hold here first.
"""

import numpy as np
import pytest

from tokengeom import train as T
from tokengeom.generative import (
    SemanticModel,
    SimulationConfig,
    de_bruijn_sequence,
    de_bruijn_successor,
)
from tokengeom.hellinger import squared_hellinger


def test_de_bruijn_covers_every_window_once():
    for k, n in ((4, 3), (6, 3), (8, 3)):
        seq = de_bruijn_sequence(k, n)
        assert len(seq) == k**n
        windows = {
            tuple(seq[(np.arange(n) + i) % len(seq)]) for i in range(len(seq))
        }
        assert len(windows) == k**n
        succ = de_bruijn_successor(k, n)
        assert succ.shape == (k**n,)
        assert set(np.unique(succ)) <= set(range(k))


def test_variant_weights_match_the_paper(sim):
    """rho_m = m^{-0.8} / sum_r r^{-0.8}, so (0.503, 0.289, 0.209) at m = 3."""
    cfg = SimulationConfig()
    full = SemanticModel(cfg)
    assert np.allclose(full.variant_weights, [0.503, 0.289, 0.209], atol=5e-4)
    assert sim.variant_weights.sum() == pytest.approx(1.0)


def test_beta_calibration_hits_the_target_quantile(sim):
    """Appendix B.1 fixes beta by matching the lower 1% quantile of the
    geometric branch at the rarest variant to 1/(n_variants |S_j(X)|)."""
    cfg = sim.config
    target = 1.0 / (cfg.n_variants * cfg.plausible_tokens)
    weights = sim._geometric_weights(sim.beta)
    values = (1.0 - cfg.exploration_mass) * weights * sim.variant_weights[-1]
    assert float(np.quantile(values.reshape(-1), 0.01)) == pytest.approx(target, rel=1e-6)
    assert sim.beta > 0.0


def test_plausible_set_shape(sim):
    """Exactly 1 + n_geometric groups per context, the exploration edge always
    distinct from the geometric slots."""
    cfg = sim.config
    counts = sim.plausible_group_mask.sum(axis=1)
    assert set(np.unique(counts)) == {cfg.plausible_groups}
    for state in range(cfg.n_groups**3):
        assert sim.successor[state] not in sim.geometric_slots[state]
    assert sim.plausible_group_mask[0, 0]  # the forced self-loop at (1,1,1)


def test_stationarity_and_irreducibility(sim):
    """p(X_{j,c}) solves the balance equation of Appendix B.1."""
    A = sim.config.n_groups
    p = sim.state_distribution
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert p.min() > 0.0
    head = (np.arange(A**3) % A**2) * A
    nxt = np.zeros_like(p)
    np.add.at(
        nxt,
        (head[:, None] + np.arange(A)[None, :]).reshape(-1),
        (p[:, None] * sim.group_conditional).reshape(-1),
    )
    assert np.abs(nxt - p).sum() < 1e-11


def test_conditional_is_a_sparse_distribution(sim):
    """Assumption 3.5: p_j(v|X) >= 1/k on its support and zero off it."""
    cond = sim.token_conditional
    assert np.allclose(cond.sum(axis=1), 1.0, atol=1e-12)
    mask = np.asarray(sim.plausible_token_mask)
    assert np.all(cond[~mask] == 0.0)
    assert cond[mask].min() >= 1.0 / sim.sparsity_k - 1e-15
    assert mask.sum(axis=1).max() == sim.config.plausible_tokens


def test_marginals(sim):
    """p(v) = rho_m p(a) and sums to one."""
    assert sim.token_marginal.sum() == pytest.approx(1.0, abs=1e-12)
    want = sim.group_marginal[sim.token_group] * sim.variant_weights[sim.token_variant]
    assert np.allclose(sim.token_marginal, want, atol=1e-15)


def test_gamma_h_is_zero_exactly_inside_a_semantic_group(sim):
    """The equivalence-collapse target of Section 2.4: exchangeable token types
    share a conditional context distribution, so Gamma_h is zero."""
    gh = np.asarray(sim.gamma_h_tokens)
    same = sim.token_group[:, None] == sim.token_group[None, :]
    assert np.abs(gh[same]).max() < 1e-14
    assert gh[~same].min() > 0.05


def test_gamma_h_agrees_with_the_observed_token_enumeration(sim):
    """Section 5 enumerates d^3 observed-token contexts; the semantic route
    over A^3 contexts has to give the same Gamma_h."""
    direct = sim.gamma_h_tokens_direct()
    assert np.abs(direct - np.asarray(sim.gamma_h_tokens)).max() < 1e-12


def test_context_conditional_is_a_distribution(sim):
    """p_v(X_{j,c}|v) from Bayes' rule, and Gamma_h read off it."""
    q = sim.group_context_conditional
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-12)
    for a1 in range(sim.config.n_groups):
        for a2 in range(sim.config.n_groups):
            got = float(squared_hellinger(q[a1], q[a2]))
            assert got == pytest.approx(sim.gamma_h_groups[a1, a2], abs=1e-12)


def test_semantic_and_token_enumerations_carry_the_same_population(sim):
    token = T.enumerate_contexts(sim)
    semantic = T.enumerate_semantic_contexts(sim)
    assert float(token.context_prob.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(semantic.context_prob.sum()) == pytest.approx(1.0, abs=1e-12)
    assert token.entropy == pytest.approx(semantic.entropy, abs=1e-12)
    lifted = T.expand_to_token_contexts(np.asarray(semantic.conditional), sim)
    assert np.abs(lifted - np.asarray(token.conditional)).max() < 1e-15


def test_sampling_matches_the_population(sim):
    """Sampled pairs reproduce p(v) and p(X_{j,c}) within Monte Carlo error."""
    contexts, targets = sim.sample_pairs(200_000, seed=7)
    counts = np.bincount(targets, minlength=sim.config.vocab_size) / len(targets)
    assert np.abs(counts - sim.token_marginal).max() < 4e-3
    states = sim.context_index(contexts)
    empirical = np.bincount(states, minlength=len(sim.state_distribution)) / len(states)
    assert np.abs(empirical - sim.state_distribution).sum() < 0.05
