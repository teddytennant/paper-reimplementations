"""Structured tensor interaction T = T_add + T_mult + T_rel (eqs 10-13)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _normal(key, shape, scale=0.1):
    return scale * jax.random.normal(key, shape, dtype=jnp.float32)


def init_mult_factor(key, d, r_k, s_dim):
    kR, kU, kV, ka = jax.random.split(key, 4)
    return {
        "R": _normal(kR, (d, r_k), scale=0.1),
        "U": _normal(kU, (r_k, d), scale=0.1),
        "V": _normal(kV, (r_k, d), scale=0.1),
        "alpha_W": _normal(ka, (s_dim,), scale=0.05),
        "alpha_b": jnp.array(0.0, dtype=jnp.float32),
        "alpha": jnp.array(1.0, dtype=jnp.float32),
    }


def init_interaction_params(key, d, s_dim, K=2, r_k=4):
    kA, kS, kF, kQ, kR = jax.random.split(key, 5)
    kAI, kAM = jax.random.split(kA)
    kSI, kSM = jax.random.split(kS)
    fkeys = jax.random.split(kF, K)
    return {
        "A_I": _normal(kAI, (d, d), scale=0.05),
        "A_M": _normal(kAM, (d, d), scale=0.05),
        "add_I_W": _normal(kSI, (d, s_dim), scale=0.05),
        "add_M_W": _normal(kSM, (d, s_dim), scale=0.05),
        "factors": tuple(init_mult_factor(fk, d, r_k, s_dim) for fk in fkeys),
        "W_q": _normal(kQ, (d, d), scale=0.1),
        "W_R": _normal(kR, (d, d), scale=0.1),
    }


def t_add(params, z_I, z_M, s=None):
    """T_add(z_I ⊗ z_M) = A_I z_I + A_M z_M, optionally FiLM-modulated by s."""
    t_I = params["A_I"] @ z_I
    t_M = params["A_M"] @ z_M
    if s is not None:
        t_I = t_I * (1.0 + jnp.tanh(params["add_I_W"] @ s))
        t_M = t_M * (1.0 + jnp.tanh(params["add_M_W"] @ s))
    return t_I + t_M


def t_mult_factor(fac, z_I, z_M, s=None):
    """One low-rank multiplicative term α R (U z_I ⊙ V z_M)."""
    if s is None:
        alpha = fac["alpha"]
    else:
        alpha = fac["alpha_W"] @ s + fac["alpha_b"]
    hadamard = (fac["U"] @ z_I) * (fac["V"] @ z_M)
    return alpha * (fac["R"] @ hadamard)


def t_mult(params, z_I, z_M, s=None):
    """T_mult = sum_k α_k R_k (U_k z_I ⊙ V_k z_M)."""
    acc = jnp.zeros_like(z_I)
    for fac in params["factors"]:
        acc = acc + t_mult_factor(fac, z_I, z_M, s=s)
    return acc


def relational_scores(params, z_M, tokens):
    """scores_ℓ = <W_q z_M, Z_I[ℓ]> / sqrt(d). tokens: (L, d)."""
    d = z_M.shape[-1]
    query = params["W_q"] @ z_M
    return (tokens @ query) / jnp.sqrt(jnp.asarray(d, dtype=z_M.dtype))


def relational_weights(params, z_M, tokens):
    scores = relational_scores(params, z_M, tokens)
    return jax.nn.softmax(scores, axis=-1)


def t_rel(params, z_M, tokens):
    """T_rel = W_R sum_ℓ π_ℓ Z_I[ℓ]. Returns (t, π). tokens=None -> 0."""
    if tokens is None:
        zeros = jnp.zeros_like(z_M)
        return zeros, None
    pi = relational_weights(params, z_M, tokens)
    pooled = pi @ tokens
    return params["W_R"] @ pooled, pi


def interaction(params, z_I, z_M, tokens=None, s=None):
    """T(z_I ⊗ z_M) = T_add + T_mult + T_rel (T_rel is 0 without tokens)."""
    t = t_add(params, z_I, z_M, s=s) + t_mult(params, z_I, z_M, s=s)
    t_r, _ = t_rel(params, z_M, tokens)
    return t + t_r
