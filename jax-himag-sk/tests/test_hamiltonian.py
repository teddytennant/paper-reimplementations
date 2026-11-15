"""SK Hamiltonian matches a hand-computed 4-spin example (eq. (10))."""

import jax
import jax.numpy as jnp
import numpy as np

from himag.sk import hamiltonian, potential, sample_goe


def test_four_spin_hand_computation():
    # J symmetric, h given. x = (+, +, −, −).
    # (Jx)_1 = 0.5 - 0.25 = 0.25
    # (Jx)_2 = 0.5 - 0.25 = 0.25
    # (Jx)_3 = 0.25 - 0.5 = -0.25
    # (Jx)_4 = 0.25 - 0.5 = -0.25
    # x^T J x = 1.0, so ½ x^T J x = 0.5
    # h^T x = 0.1 - 0.2 - 0.3 + 0.4 = 0.0
    J = jnp.asarray(
        [
            [0.0, 0.5, 0.25, 0.0],
            [0.5, 0.0, 0.0, 0.25],
            [0.25, 0.0, 0.0, 0.5],
            [0.0, 0.25, 0.5, 0.0],
        ]
    )
    h = jnp.asarray([0.1, -0.2, 0.3, -0.4])
    x = jnp.asarray([1.0, 1.0, -1.0, -1.0])
    f = potential(x, J, h)
    H = hamiltonian(x, J, h)
    assert abs(float(f) - 0.5) < 1e-6
    assert abs(float(H) - (-0.5)) < 1e-6
    assert np.isfinite(float(f)) and np.isfinite(float(H))


def test_goe_variances_and_symmetry():
    d = 40
    key = jax.random.PRNGKey(2)
    J = np.asarray(sample_goe(key, d))
    assert np.allclose(J, J.T, atol=1e-6)
    off = J[np.triu_indices(d, 1)]
    diag = np.diag(J)
    # Model 1: off-diag Var = 1/d, diag Var = 2/d. Generous on finite-d.
    assert abs(off.var() - 1.0 / d) < 0.02
    assert abs(diag.var() - 2.0 / d) < 0.08
    assert np.all(np.isfinite(J))


def test_potential_batched_and_finite():
    d = 5
    J = sample_goe(jax.random.PRNGKey(3), d)
    h = jnp.arange(d, dtype=jnp.float32) * 0.1
    xs = jnp.asarray(
        [
            [1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0, -1.0],
        ]
    )
    f = potential(xs, J, h)
    assert f.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(f)))
    assert bool(jnp.all(jnp.isfinite(hamiltonian(xs, J, h))))
