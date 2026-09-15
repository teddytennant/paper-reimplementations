"""Tiny synthetic two-community graphs. No Planetoid / OGB downloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def two_community_graph(
    key,
    n_per=8,
    feat_dim=6,
    p_in=0.8,
    p_out=0.05,
    noise=0.15,
):
    """Homophilous stochastic block model with community labels.

    Features are one-hot community indicators plus Gaussian noise, so labels
    are a function of community and are linearly readable from X.
    """
    n = 2 * n_per
    y = jnp.concatenate(
        [
            jnp.zeros((n_per,), dtype=jnp.int32),
            jnp.ones((n_per,), dtype=jnp.int32),
        ]
    )
    k_x, k_a = jax.random.split(key)
    proto = jax.nn.one_hot(y, feat_dim)
    x = proto + noise * jax.random.normal(k_x, (n, feat_dim))
    same = y[:, None] == y[None, :]
    p = jnp.where(same, p_in, p_out)
    u = jax.random.uniform(k_a, (n, n))
    upper = jnp.triu((u < p).astype(jnp.float32), k=1)
    adj = jnp.maximum(upper, upper.T)
    return x.astype(jnp.float32), adj, y
