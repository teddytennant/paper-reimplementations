"""Implicit VJP on a linear contraction matches (I-A)^{-T}."""

import jax
import jax.numpy as jnp

from q_deq.implicit import implicit_adjoint


def test_implicit_grad_matches_closed_form():
    d, din = 7, 5
    key = jax.random.PRNGKey(12)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # Strict contraction: spectral radius 0.4.
    U = jax.random.normal(k1, (d, d))
    Qmat, _ = jnp.linalg.qr(U)
    A = 0.4 * Qmat
    B = 0.3 * jax.random.normal(k2, (d, din))
    x = jax.random.normal(k3, (din,))
    g = jax.random.normal(k4, (d,))

    def F(z, xx):
        return A @ z + B @ xx

    I = jnp.eye(d)
    z_star = jnp.linalg.solve(I - A, B @ x)
    # Fixed point check.
    assert jnp.allclose(F(z_star, x), z_star, atol=1e-5)

    w = implicit_adjoint(F, z_star, x, g)
    w_closed = jnp.linalg.solve((I - A).T, g)
    assert jnp.allclose(w, w_closed, atol=1e-5, rtol=1e-5)

    # One-step Neumann I + A^T is not enough.
    w_neumann = g + A.T @ g
    assert jnp.linalg.norm(w_neumann - w_closed) > 1e-3
