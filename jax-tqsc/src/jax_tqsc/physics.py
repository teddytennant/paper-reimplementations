"""Physical output map (S46–S51) and metrics (S52 / S63)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS = 1e-6


def _softplus_diag(r: jax.Array, eps: float = EPS) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    s0 = jax.nn.softplus(r[..., 0]) + eps
    s1 = jax.nn.softplus(r[..., 1]) + eps
    return s0, s1, r[..., 2], r[..., 3]


def rho_from_chol(r: jax.Array, eps: float = EPS) -> jax.Array:
    """Lower-triangular Cholesky construction (S46–S50): rho = C C† / Tr(C C†)."""
    s0, s1, r2, r3 = _softplus_diag(r, eps)
    zero = jnp.zeros_like(s0)
    c00 = jax.lax.complex(s0, zero)
    c01 = jax.lax.complex(zero, zero)
    c10 = jax.lax.complex(r2, r3)
    c11 = jax.lax.complex(s1, zero)
    row0 = jnp.stack([c00, c01], axis=-1)
    row1 = jnp.stack([c10, c11], axis=-1)
    c = jnp.stack([row0, row1], axis=-2)
    c_dag = jnp.conjugate(jnp.moveaxis(c, -1, -2))
    cc = c @ c_dag
    tr = jnp.real(jnp.trace(cc, axis1=-2, axis2=-1))
    return cc / tr[..., None, None]


def cholesky_rho(r: jax.Array, eps: float = EPS) -> jax.Array:
    """Closed form (S51) equivalent to C C† / Tr(C C†) for d=2."""
    s0, s1, r2, r3 = _softplus_diag(r, eps)
    nrm = s0 * s0 + s1 * s1 + r2 * r2 + r3 * r3
    rho00 = (s0 * s0) / nrm
    rho11 = (s1 * s1 + r2 * r2 + r3 * r3) / nrm
    rho01 = jax.lax.complex(s0 * r2, -s0 * r3) / nrm
    rho10 = jax.lax.complex(s0 * r2, s0 * r3) / nrm
    row0 = jnp.stack([rho00.astype(rho01.dtype), rho01], axis=-1)
    row1 = jnp.stack([rho10, rho11.astype(rho10.dtype)], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def frobenius_mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared Frobenius error (S52 / S63)."""
    diff = pred - target
    sq = jnp.real(diff * jnp.conjugate(diff))
    return jnp.mean(jnp.sum(sq, axis=(-2, -1)))


def fidelity(rho: jax.Array, sigma: jax.Array) -> jax.Array:
    """Squared Uhlmann fidelity. For qubits this is Jozsa's formula.

    F = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2
      = Tr(rho sigma) + 2 sqrt(det(rho) det(sigma))  (d=2).
    """
    overlap = jnp.real(jnp.trace(rho @ sigma, axis1=-2, axis2=-1))
    d1 = jnp.clip(jnp.real(jnp.linalg.det(rho)), 0.0, None)
    d2 = jnp.clip(jnp.real(jnp.linalg.det(sigma)), 0.0, None)
    return overlap + 2.0 * jnp.sqrt(d1 * d2)


def maximally_mixed(batch: int | None = None) -> jax.Array:
    eye = jnp.eye(2, dtype=jnp.complex64) * 0.5
    if batch is None:
        return eye
    return jnp.broadcast_to(eye, (batch, 2, 2))
