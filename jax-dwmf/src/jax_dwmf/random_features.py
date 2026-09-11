"""Tiny optimized-random-feature sampler (Theorem 2 / Algorithm 2, small n).

Uses Algorithm 1 with D = sqrt(q̂^{(ρ)}), A = k, γ = ϵ, f(x) = x^{-1}
(Eqs. (1), (22), (D11)). Kernel entries follow the cosine form (E5).

Not implemented: QSVT, the full G^D grid for D > 1, and the paper's
I-step rejection sampler with uniform-fallback (Alg. 2 steps 3–17).
On a small 1-D grid we evaluate w̃(x) (E26) exactly and sample from
the normalized nonnegative weights.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from .algorithm import SparseMatrixFunction, diagonal_truncation_transform, materialize_N

Array = jax.Array


def cosine_kernel_entry(Q_tau: Array, alpha: int, beta: int) -> Array:
    """Exact k(α, β) = G^{-1} Σ_y Q^{(τ)}(y) cos(2π y (α−β) / G) (Eq. (E5))."""
    G = Q_tau.shape[0]
    y = jnp.arange(G)
    phase = 2.0 * jnp.pi * y * (alpha - beta) / G
    return jnp.sum(Q_tau * jnp.cos(phase)) / G


def shift_invariant_kernel(Q_tau: Array) -> Array:
    """Dense 1-D kernel k = F† Q^{(τ)} F via Eq. (E5)."""
    G = int(Q_tau.shape[0])
    idx = jnp.arange(G)
    delta = idx[:, None] - idx[None, :]
    y = jnp.arange(G)
    phase = 2.0 * jnp.pi * y[None, None, :] * delta[:, :, None] / G
    return jnp.sum(Q_tau[None, None, :] * jnp.cos(phase), axis=-1) / G


def estimate_kernel_entry(
    key: jax.Array,
    Q_tau: Array,
    alpha: int,
    beta: int,
    eps: float,
    delta: float,
    Q_max: Optional[float] = None,
) -> Array:
    """Lemma 20: Monte Carlo estimate of k(α, β) by uniform samples over X.

    I = ceil( 2 (Q_max / ϵ)^2 ln(2/δ) ). Default path for small G is
    ``cosine_kernel_entry`` (exact).
    """
    G = Q_tau.shape[0]
    if Q_max is None:
        Q_max = float(jnp.max(Q_tau))
    I = int(jnp.ceil(2.0 * (Q_max / eps) ** 2 * jnp.log(2.0 / delta)))
    I = max(I, 1)
    y = jax.random.randint(key, (I,), 0, G)
    phase = 2.0 * jnp.pi * y * (alpha - beta) / G
    return jnp.mean(Q_tau[y] * jnp.cos(phase))


def _f_inv(x: Array) -> Array:
    return 1.0 / x


def approximate_sigma_inv(
    key: jax.Array,
    q_hat: Array,
    k: Array,
    eps_reg: float,
    *,
    eps_D: float,
    delta_D: float = 1e-6,
    exact_support: bool = True,
) -> SparseMatrixFunction:
    """Algorithm 1 on Σ_ϵ = ϵ I + sqrt(q̂) k sqrt(q̂) with f(x) = x^{-1}."""
    d = jnp.sqrt(jnp.clip(q_hat, 0.0))
    return diagonal_truncation_transform(
        key,
        d,
        k,
        gamma=eps_reg,
        f=_f_inv,
        eps_D=eps_D,
        delta_D=delta_D,
        exact_support=exact_support,
    )


def dft_matrix(G: int, dtype=jnp.complex128) -> Array:
    """1-D unitary DFT, F(x, y) = G^{-1/2} exp(−2π i x y / G) (Eq. (D1))."""
    x = jnp.arange(G)
    omega = jnp.exp(-2j * jnp.pi * x[:, None] * x[None, :] / G)
    return omega.astype(dtype) / jnp.sqrt(G)


def optimized_feature_weights(
    q_hat: Array,
    Q_tau: Array,
    N: Array,
) -> Array:
    """w̃(x) = ⟨φ_x | N | φ_x⟩ with |φ_x⟩ = sqrt(q̂) F sqrt(Q^{(τ)}) |x⟩ (E26).

    For the 1-D grid this is Q(x) (sqrt(q̂) ⊙ F[:, x])^* N (sqrt(q̂) ⊙ F[:, x]).
    """
    G = int(q_hat.shape[0])
    F = dft_matrix(G)
    sqrt_q = jnp.sqrt(jnp.clip(q_hat, 0.0)).astype(N.dtype)
    N_c = N.astype(F.dtype)
    sqrt_q_c = sqrt_q.astype(F.dtype)

    def one(x: Array) -> Array:
        u = sqrt_q_c * F[:, x]
        quad = jnp.vdot(u, N_c @ u)
        return (Q_tau[x] * quad.real).astype(N.dtype)

    return jax.vmap(one)(jnp.arange(G))


def sample_optimized_features(
    key: jax.Array,
    q_hat: Array,
    Q_tau: Array,
    eps_reg: float,
    *,
    eps_D: float,
    n_samples: int = 1,
    exact_support: bool = True,
    k: Optional[Array] = None,
) -> tuple[Array, Array, SparseMatrixFunction]:
    """Tiny classical ORF sampler.

    Returns ``(samples, weights, N_repr)`` where samples are drawn from
    the normalized nonnegative part of w̃ (Definition 18 / Eq. (E25)
    with N in place of Σ_ϵ^{-1}).
    """
    if k is None:
        k = shift_invariant_kernel(Q_tau)
    repr_N = approximate_sigma_inv(
        key,
        q_hat,
        k,
        eps_reg,
        eps_D=eps_D,
        exact_support=exact_support,
    )
    N = materialize_N(repr_N)
    w = optimized_feature_weights(q_hat, Q_tau, N)
    w_clip = jnp.clip(w, 0.0)
    total = jnp.sum(w_clip)
    p = jnp.where(
        total > 0,
        w_clip / total,
        jnp.ones_like(w_clip) / w_clip.shape[0],
    )
    samples = jax.random.choice(key, q_hat.shape[0], shape=(n_samples,), p=p)
    return samples, w, repr_N
