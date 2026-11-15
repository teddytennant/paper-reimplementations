"""Sherrington–Kirkpatrick Hamiltonian on the Hamming slice.

Gibbs measure (paper eq. (10)):
    pi(x) ∝ exp(β (½ x^T J x + h^T x)) on {±1}^d,
restricted to X_k^d for fixed-magnetization sampling.

Model 1 (GOE): J_ij ~ N(0, 1/d) for i < j, J_ii ~ N(0, 2/d), J symmetric.
Off-diagonal std 1/sqrt(d) is the paper's SK scaling. The diagonal is
constant on the hypercube (x_i^2 = 1) and does not affect pi.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def _as_field(h, d: int, dtype) -> Array:
    h = jnp.asarray(h, dtype=dtype)
    if h.ndim == 0:
        return jnp.full((d,), h)
    return h


def sample_goe(key: Array, d: int, dtype=jnp.float32) -> Array:
    """Draw J ~ GOE(d) as in Model 1: off-diag N(0, 1/d), diag N(0, 2/d)."""
    g = jax.random.normal(key, (d, d), dtype=dtype)
    # (G + G^T)/sqrt(2) has off-diag N(0, 1) and diag N(0, 2); then /sqrt(d).
    wigner = (g + g.T) / jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
    return wigner / jnp.sqrt(jnp.asarray(d, dtype=dtype))


def potential(x: Array, J: Array, h) -> Array:
    """f(x) = ½ x^T J x + h^T x, paper eq. (10) and Section 5.3."""
    x = jnp.asarray(x)
    J = jnp.asarray(J)
    h = _as_field(h, x.shape[-1], x.dtype)
    quad = 0.5 * jnp.einsum("...i,ij,...j->...", x, J, x)
    lin = jnp.einsum("...i,i->...", x, h)
    return quad + lin


def hamiltonian(x: Array, J: Array, h) -> Array:
    """SK Hamiltonian H(x) = -f(x), so pi ∝ exp(-β H) = exp(β f)."""
    return -potential(x, J, h)


def log_unnormalized(x: Array, J: Array, h, beta: float) -> Array:
    """log pi(x) up to a constant: β f(x)."""
    return jnp.asarray(beta, dtype=jnp.asarray(x).dtype) * potential(x, J, h)


def boltzmann_probs(xs: Array, J: Array, h, beta: float) -> Array:
    """Exact Boltzmann probabilities on a finite list of spin vectors."""
    logw = log_unnormalized(xs, J, h, beta)
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    return w / jnp.sum(w)
