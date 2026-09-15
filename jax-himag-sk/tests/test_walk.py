"""Transition probabilities: nonnegative, sum to 1; Remark 2 matches π."""

import jax
import jax.numpy as jnp
import numpy as np

from himag.sk import potential, sample_goe
from himag.slice import enumerate_slice, random_slice_spin, spins_from_index_set
from himag.walk import down_up_step, transition_matrix, up_log_weights


def test_transition_nonnegative_and_stochastic():
    d, k, beta = 5, 2, 0.8
    J = sample_goe(jax.random.PRNGKey(4), d)
    h = jax.random.normal(jax.random.PRNGKey(5), (d,))
    states, P = transition_matrix(J, h, beta, d, k)
    P = np.asarray(P)
    assert np.all(P >= -1e-12)
    assert np.all(np.isfinite(P))
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert states.shape[0] == 10  # C(5, 2)


def test_remark2_matches_brute_force_potential():
    d, beta = 6, 1.1
    J = sample_goe(jax.random.PRNGKey(6), d)
    h = jax.random.normal(jax.random.PRNGKey(7), (d,))
    # T = {0, 2}  (k-1 = 2, so k = 3)
    t_mask = jnp.array([True, False, True, False, False, False])
    log_w = np.asarray(up_log_weights(t_mask, J, h, beta))
    brute = np.empty(d)
    for i in range(d):
        idx = [0, 2] if i in (0, 2) else [0, 2, i]
        x = spins_from_index_set(idx, d)
        brute[i] = float(jnp.asarray(beta) * potential(x, J, h))
    cand = ~np.asarray(t_mask)
    # Remark 2 and β f differ by an i-independent constant on i ∉ T.
    diff = log_w[cand] - brute[cand]
    np.testing.assert_allclose(diff, diff.mean(), atol=1e-4)


def test_step_finite():
    d, k = 7, 2
    J = sample_goe(jax.random.PRNGKey(8), d)
    h = jnp.ones((d,))
    x = random_slice_spin(jax.random.PRNGKey(9), d, k)
    y = down_up_step(jax.random.PRNGKey(10), x, J, h, 0.5)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert int(jnp.sum(y > 0)) == k
