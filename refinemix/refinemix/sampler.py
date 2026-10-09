"""Ancestral sampler for the masked kernel.

A reverse step from time t to s < t uses equation 12 with the denoiser in
place of z_0, under the SUBS constraint:
    unmasked tokens are copied,
    a masked token stays masked with probability (1 - alpha_s) / (1 - alpha_t)
    and otherwise is drawn from the denoiser (which puts zero mass on the mask).

The final step has s = 0, so alpha_s = 1 and the stay-mask probability is 0:
the mask is never emitted at t = 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from refinemix.loss import subs_logits
from refinemix.schedule import EPS, alpha


def masked_reverse_sample(key, tokens, logits, alpha_s, alpha_t, mask_id: int):
    """One ancestral step. `logits` are (B, L, V); mask logit should be -inf."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    logits = subs_logits(logits, mask_id)
    a_s = jnp.asarray(alpha_s, dtype=jnp.float32)
    a_t = jnp.asarray(alpha_t, dtype=jnp.float32)
    stay_prob = (1.0 - a_s) / (1.0 - a_t)
    stay_prob = jnp.clip(stay_prob, 0.0, 1.0)
    is_mask = tokens == int(mask_id)
    key_u, key_cat = jax.random.split(key)
    u = jax.random.uniform(key_u, tokens.shape)
    sampled = jax.random.categorical(key_cat, logits, axis=-1).astype(jnp.int32)
    revealed = jnp.where(u < stay_prob, jnp.int32(mask_id), sampled)
    return jnp.where(is_mask, revealed, tokens)


def ancestral_sample(
    apply_fn,
    params,
    key,
    *,
    batch_size: int,
    seq_len: int,
    n_steps: int,
    mask_id: int,
    eps: float = EPS,
    ideal: bool = False,
):
    """Start from all masks at t = 1 and ancestral-sample down to t = 0.

    Returns integer tokens (batch_size, seq_len) with no mask tokens.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    @jax.jit
    def _run(params, key):
        times = jnp.linspace(1.0, 0.0, n_steps + 1)
        x = jnp.full((batch_size, seq_len), int(mask_id), dtype=jnp.int32)
        keys = jax.random.split(key, n_steps)

        def step(x, inputs):
            t, s, k = inputs
            logits = apply_fn(params, x, t)
            a_t = alpha(t, eps=eps, ideal=ideal)
            a_s = alpha(s, eps=eps, ideal=ideal)
            x = masked_reverse_sample(k, x, logits, a_s, a_t, mask_id)
            return x, None

        x, _ = jax.lax.scan(step, x, (times[:-1], times[1:], keys))
        return x

    return _run(params, key)
