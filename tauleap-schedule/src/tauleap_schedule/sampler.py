"""Tau-leap unmasking (Alg. 1 / Alg. 2) on the toy alphabet {0, 1, MASK}^N."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

MASK = 2


def product_bernoulli_predictor(p: float = 0.5) -> Callable:
    """Independent-bit predictor p_θ used in tests (not a trained MDM)."""

    def predict(x, mask, key):
        bits = jax.random.bernoulli(key, p, shape=x.shape)
        return bits.astype(x.dtype)

    return predict


def tau_leap_alg1(x, mask, q, key, predictor):
    """Reveal each still-masked coordinate independently with probability q_k."""
    key_reveal, key_pred = jax.random.split(key)
    reveal = jax.random.bernoulli(key_reveal, q, shape=mask.shape) & mask
    samples = predictor(x, mask, key_pred)
    x_new = jnp.where(reveal, samples, x)
    mask_new = mask & ~reveal
    return x_new, mask_new


def tau_leap_alg2(x, mask, q, key, predictor):
    """Draw s ~ Binomial(n_masked, q_k) and reveal a uniform subset of size s."""
    key_s, key_pick, key_pred = jax.random.split(key, 3)
    bern = jax.random.bernoulli(key_s, q, shape=mask.shape)
    s = jnp.sum(bern & mask)
    noise = jax.random.uniform(key_pick, shape=x.shape)
    noise = jnp.where(mask, noise, jnp.array(-1.0, dtype=noise.dtype))
    rank = jnp.argsort(jnp.argsort(-noise))
    reveal = mask & (rank < s)
    samples = predictor(x, mask, key_pred)
    x_new = jnp.where(reveal, samples, x)
    mask_new = mask & ~reveal
    return x_new, mask_new


def run_tau_leap(alg, x, mask, q, key, predictor):
    """Apply ``alg`` for each q_k in order."""
    q = jnp.asarray(q)
    n_steps = int(q.shape[0])

    def body(i, carry):
        x_i, mask_i, key_i = carry
        key_i, k_step = jax.random.split(key_i)
        x_i, mask_i = alg(x_i, mask_i, q[i], k_step, predictor)
        return x_i, mask_i, key_i

    x, mask, _ = jax.lax.fori_loop(0, n_steps, body, (x, mask, key))
    return x, mask
