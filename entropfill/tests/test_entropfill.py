"""Invariants from Propositions 3.1-3.2 and Theorem 4.1."""

import jax
import jax.numpy as jnp
import numpy as np

from entropfill.envelope import chi2, envelope_holds
from entropfill.pooling import pool
from entropfill.scan import chunk_scores, feasible_scan
from entropfill.weights import head_weights


def _attention(key, groups, heads, queries, positions):
    raw = jax.random.uniform(key, (groups, heads, queries, positions))
    return raw / jnp.sum(raw, axis=-1, keepdims=True)


def test_pooled_distribution_is_a_probability():
    attention = _attention(jax.random.PRNGKey(0), 2, 3, 4, 12)
    sink = jnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    weights, _ = head_weights(attention, sink, z_min=0.05, tau=1.0, zeta=0.1)
    mu = pool(attention, weights)
    assert abs(float(jnp.sum(mu)) - 1.0) < 1e-5
    assert jnp.all(mu >= -1e-8)


def test_kappa_is_between_zero_and_one():
    positions = 20
    pooled = jnp.ones(positions) / positions
    chunks = jnp.zeros((4, positions))
    chunks = chunks.at[0, 2:5].set(1.0)
    chunks = chunks.at[1, 5:12].set(1.0)
    chunks = chunks.at[2, 12:13].set(1.0)
    chunks = chunks.at[3, 13:20].set(1.0)
    document = jnp.ones(positions)
    _, kappa, _ = chunk_scores(pooled, document, chunks, epsilon_s=0.1, lam=0.5)
    assert jnp.all(kappa >= -1e-6)
    assert jnp.all(kappa <= 1.0 + 1e-6)


def test_uniform_document_attention_scores_zero():
    positions = 16
    pooled = jnp.ones(positions) / positions
    chunks = jnp.zeros((4, positions))
    for i, span in enumerate((slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16))):
        chunks = chunks.at[i, span].set(1.0)
    _, _, scores = chunk_scores(pooled, jnp.ones(positions), chunks, 0.05, 0.3)
    assert jnp.max(jnp.abs(scores)) < 1e-5


def test_envelope_holds_on_random_rows():
    key = jax.random.PRNGKey(1)
    positions = 16
    row = jax.random.dirichlet(key, jnp.ones(positions))
    other = jax.random.dirichlet(jax.random.fold_in(key, 1), jnp.ones(positions))
    omega = 0.3
    pooled = omega * row + (1.0 - omega) * other
    removal = jnp.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    assert bool(envelope_holds(row, pooled, omega, removal))


def test_sharp_envelope_is_tight():
    # The paper's sharpness example: a puts all mass on one position, b is
    # supported off it, mu = omega a + (1-omega) b. Both nontrivial terms
    # equal one, so the envelope is one and the row mass is one.
    omega = 0.25
    row = jnp.array([1.0, 0.0, 0.0, 0.0])
    other = jnp.array([0.0, 0.5, 0.5, 0.0])
    pooled = omega * row + (1.0 - omega) * other
    removal = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert abs(float(chi2(row, pooled)) - (1.0 / omega - 1.0)) < 1e-5
    assert bool(envelope_holds(row, pooled, omega, removal))


def test_scan_respects_the_removal_budget():
    scores = jnp.array([0.1, 0.4, 0.2, 0.9])
    lengths = jnp.array([3.0, 5.0, 4.0, 2.0])
    row_mass = jnp.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.5, 0.2, 0.2]])
    removed = feasible_scan(
        scores, lengths, row_mass, row_budget=jnp.array([0.5, 0.5]), max_removed=2, token_target=20
    )
    assert int(jnp.sum(removed)) <= 2
    spent = row_mass @ removed.astype(float)
    assert jnp.all(spent <= 0.5 + 1e-6)


def test_mass_advantage_beats_concentration():
    # Proposition 3.2: ell_i / ell_j > exp(lambda) implies S_i > S_j.
    positions = 10
    pooled = jnp.array([0.4, 0.4, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04])
    chunks = jnp.zeros((2, positions))
    chunks = chunks.at[0, 0:2].set(1.0)
    chunks = chunks.at[1, 2:10].set(1.0)
    lam = 0.5
    ell, _, scores = chunk_scores(pooled, jnp.ones(positions), chunks, 1e-3, lam)
    assert float(ell[0] / ell[1]) > np.exp(lam)
    assert float(scores[0]) > float(scores[1])
