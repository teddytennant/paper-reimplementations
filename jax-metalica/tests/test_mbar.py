"""MBAR weights and the uniform-bias special case."""

import jax.numpy as jnp
import numpy as np

from jax_metalica.mbar import free_energy_difference, mbar_weights, solve_mbar


def test_mbar_weights_nonneg_sum_to_one():
    n_states, n_samples = 3, 12
    V = jnp.linspace(-1.0, 1.0, n_states * n_samples).reshape(n_states, n_samples)
    n_counts = jnp.array([4.0, 4.0, 4.0])
    w = mbar_weights(V, n_counts)
    assert w.shape == (n_samples,)
    assert jnp.all(w >= 0.0)
    np.testing.assert_allclose(float(w.sum()), 1.0, atol=1e-5)


def test_mbar_uniform_when_all_V_zero():
    n_states, n_samples = 4, 20
    V = jnp.zeros((n_states, n_samples))
    n_counts = jnp.full((n_states,), 5.0)
    f = solve_mbar(V, n_counts)
    w = mbar_weights(V, n_counts, f=f)
    np.testing.assert_allclose(w, jnp.full((n_samples,), 1.0 / n_samples), atol=1e-5)


def test_delta_F_sign_when_mass_on_negative_side():
    x = jnp.array([-2.0, -1.0, -0.5, 0.2])
    w = jnp.array([0.4, 0.4, 0.15, 0.05])
    dF = float(free_energy_difference(x, w))
    assert np.isfinite(dF)
    assert dF > 0.0
