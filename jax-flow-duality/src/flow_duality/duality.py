"""Monte Carlo induced discrete coefficient from argmax projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flow_duality.interpolant import (
    argmax_tokens,
    convex_interpolant,
    one_hot_tokens,
    sample_distinct_token,
)
from flow_duality.sources import sample_source


def induced_coefficient_mc(
    key,
    k_tilde,
    vocab_size,
    *,
    family="gaussian",
    n_samples=8000,
    z0=None,
    width=1.0,
    low=0.0,
    beta=1.0,
):
    """Estimate k_t = P(argmax(Z_t) = x1 | x0 != x1).

    Samples Z_0 from the continuous source (or uses ``z0``), draws a one-hot
    target Z_1 with x1 != x0, forms Z_t = (1 - k_tilde) Z_0 + k_tilde Z_1,
    and returns the fraction of coordinates whose min-index argmax equals x1.
    """
    vocab_size = int(vocab_size)
    if z0 is None:
        key, sk = jax.random.split(key)
        z0 = sample_source(
            sk,
            (n_samples, vocab_size),
            family,
            width=width,
            low=low,
            beta=beta,
        )
    key, sk = jax.random.split(key)
    x0 = argmax_tokens(z0)
    x1 = sample_distinct_token(sk, x0, vocab_size)
    z1 = one_hot_tokens(x1, vocab_size, dtype=z0.dtype)
    zt = convex_interpolant(z0, z1, k_tilde)
    xt = argmax_tokens(zt)
    return jnp.mean(xt == x1)
