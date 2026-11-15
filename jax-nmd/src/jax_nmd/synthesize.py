"""Pole-residue synthesis (eq 2, 4) and Z to S (section 3.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _j_omega(omega):
    """j * omega with the dtype of `omega` (avoid Python 1j promoting to c128)."""
    omega = jnp.asarray(omega)
    return jax.lax.complex(jnp.zeros_like(omega), omega)


def synthesize_z(omega, poles, left, right):
    """Zij(ω) = sum_k l_{k,i} r_{k,j} / (jω - p_k).

    omega: (F,) real angular frequencies.
    poles: (K,) or (B, K) complex.
    left:  (K, N) or (B, K, N) complex left couplings l_{k,i}.
    right: (K, N) or (B, K, N) complex right couplings r_{k,j}.
    returns Z: (F, N, N) or (B, F, N, N).
    """
    omega = jnp.asarray(omega)
    poles = jnp.asarray(poles)
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    unbatched = poles.ndim == 1
    if unbatched:
        poles = poles[None, ...]
        left = left[None, ...]
        right = right[None, ...]
    jw = _j_omega(omega)
    denom = jw[None, :, None] - poles[:, None, :]
    inv = 1.0 / denom
    z = jnp.einsum("bki,bkj,bfk->bfij", left, right, inv)
    if unbatched:
        z = z[0]
    return z


def z_to_s(z, z0=50.0):
    """S = (Z - Z0 I)(Z + Z0 I)^{-1}, section 3.1.

    z: (..., N, N) complex. z0 default 50 ohms.
    """
    z = jnp.asarray(z)
    n = z.shape[-1]
    eye = jnp.eye(n, dtype=z.dtype)
    z0c = jnp.asarray(z0, dtype=z.dtype)
    z0i = z0c * eye
    return (z - z0i) @ jnp.linalg.inv(z + z0i)


def dominance_score(omega, poles, left, right):
    """Appendix C eq 18, generalized to N ports.

    For two ports, ||M_k||_F = sqrt((|l_i|^2+|l_j|^2)(|r_i|^2+|r_j|^2)),
    which equals ||l_k||_2 ||r_k||_2. The kernel gain is
    sqrt(sum_n 1/|jω_n - p_k|^2).
    """
    omega = jnp.asarray(omega)
    poles = jnp.asarray(poles)
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    mk = jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1)
    jw = _j_omega(omega)
    dist = jnp.abs(jw[:, None] - poles[None, :])
    gain = jnp.sqrt(jnp.sum(1.0 / (dist * dist), axis=0))
    return mk * gain
