"""Detailed balance of Algorithm 1 on an enumerable 3-state restriction."""

import jax
import numpy as np

from himag.sk import boltzmann_probs, sample_goe
from himag.walk import transition_matrix


def test_detailed_balance_d3_k2():
    # X_2^3 has three states: ++−, +−+, −++. Fully enumerable.
    d, k, beta = 3, 2, 1.3
    J = sample_goe(jax.random.PRNGKey(11), d)
    h = jax.random.normal(jax.random.PRNGKey(12), (d,))
    states, P = transition_matrix(J, h, beta, d, k)
    pi = np.asarray(boltzmann_probs(states, J, h, beta))
    P = np.asarray(P)
    flow = pi[:, None] * P
    np.testing.assert_allclose(flow, flow.T, atol=1e-5)
    assert np.all(np.isfinite(flow))


def test_detailed_balance_d4_k2():
    d, k, beta = 4, 2, 0.6
    J = sample_goe(jax.random.PRNGKey(13), d)
    h = jnp_h(d)
    states, P = transition_matrix(J, h, beta, d, k)
    pi = np.asarray(boltzmann_probs(states, J, h, beta))
    flow = pi[:, None] * np.asarray(P)
    np.testing.assert_allclose(flow, flow.T, atol=1e-5)


def jnp_h(d):
    return jax.random.normal(jax.random.PRNGKey(14), (d,))
