"""Reverse posteriors, equations 11 and 12.

Equation 11 is the Bayes posterior of the interpolating chain shared by both
kernels, with alpha_{t|s} = alpha_t / alpha_s:

    p(z_s | z_t, z_0) = Cat( z_s ;
        [alpha_{t|s} z_t + (1 - alpha_{t|s}) <z_t, pi> 1]
        odot [alpha_s z_0 + (1 - alpha_s) pi]
        / [alpha_t <z_t, z_0> + (1 - alpha_t) <z_t, pi>] )

Equation 12 is the masked-kernel special case (pi = point mass on the mask):
    if z_t != mask: point mass on z_t
    if z_t == mask: Cat( ((1 - alpha_s) mask + (alpha_s - alpha_t) z_0) / (1 - alpha_t) )

`bayes_posterior_probs` evaluates the same posterior by normalizing
p(z_t | z_s) p(z_s | z_0), where each factor is equation 1 (the transition
uses alpha_{t|s}).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from refinemix.kernels import forward_probs


def masked_posterior_probs(zt, z0, alpha_s, alpha_t, mask_id: int, vocab_size: int):
    """Equation 12 as a probability tensor of shape zt.shape + (vocab_size,)."""
    zt = jnp.asarray(zt, dtype=jnp.int32)
    z0 = jnp.asarray(z0, dtype=jnp.int32)
    dtype = jnp.float32
    a_s = jnp.broadcast_to(jnp.asarray(alpha_s, dtype=dtype), zt.shape)
    a_t = jnp.broadcast_to(jnp.asarray(alpha_t, dtype=dtype), zt.shape)
    v = int(vocab_size)
    mask_oh = jax.nn.one_hot(jnp.full(zt.shape, int(mask_id), dtype=jnp.int32), v, dtype=dtype)
    z0_oh = jax.nn.one_hot(z0, v, dtype=dtype)
    denom = (1.0 - a_t)[..., None]
    masked = ((1.0 - a_s)[..., None] * mask_oh + (a_s - a_t)[..., None] * z0_oh) / denom
    copied = jax.nn.one_hot(zt, v, dtype=dtype)
    is_mask = (zt == int(mask_id))[..., None]
    return jnp.where(is_mask, masked, copied)


def general_posterior_probs(zt, z0, alpha_s, alpha_t, pi):
    """Equation 11 for one token. `zt`, `z0` are integer indices. Returns (V,)."""
    pi = jnp.asarray(pi, dtype=jnp.float32)
    v = pi.shape[0]
    zt_i = int(zt)
    z0_i = int(z0)
    a_s = float(alpha_s)
    a_t = float(alpha_t)
    if a_s == 0.0:
        raise ValueError("alpha_s = 0 makes alpha_{t|s} undefined")
    a_ts = a_t / a_s
    zt_oh = jnp.zeros((v,), dtype=jnp.float32).at[zt_i].set(1.0)
    z0_oh = jnp.zeros((v,), dtype=jnp.float32).at[z0_i].set(1.0)
    dot_zt_pi = pi[zt_i]
    dot_zt_z0 = jnp.asarray(1.0 if zt_i == z0_i else 0.0, dtype=jnp.float32)
    left = a_ts * zt_oh + (1.0 - a_ts) * dot_zt_pi * jnp.ones((v,), dtype=jnp.float32)
    right = a_s * z0_oh + (1.0 - a_s) * pi
    numer = left * right
    denom = a_t * dot_zt_z0 + (1.0 - a_t) * dot_zt_pi
    return numer / denom


def bayes_posterior_probs(zt, z0, alpha_s, alpha_t, pi):
    """Direct Bayes posterior from equation 1.

    p(z_s | z_t, z_0) = p(z_t | z_s) p(z_s | z_0) / p(z_t | z_0),
    with p(z_t | z_s) = equation 1 at alpha_{t|s} = alpha_t / alpha_s.
    """
    pi = jnp.asarray(pi, dtype=jnp.float32)
    v = int(pi.shape[0])
    zt_i = int(zt)
    z0_i = int(z0)
    a_s = float(alpha_s)
    a_t = float(alpha_t)
    a_ts = a_t / a_s
    p_zs = forward_probs(jnp.asarray(z0_i), jnp.asarray(a_s), pi)
    p_zt = forward_probs(jnp.asarray(z0_i), jnp.asarray(a_t), pi)[zt_i]
    zs_index = jnp.arange(v)
    # p(z_t | z_s) for every z_s, via equation 1.
    trans = forward_probs(zs_index, jnp.asarray(a_ts), pi)[:, zt_i]
    joint = trans * p_zs
    return joint / p_zt
