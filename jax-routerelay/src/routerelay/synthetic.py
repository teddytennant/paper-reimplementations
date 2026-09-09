"""Scaled-down section 5.1 drift process.

Each query row keeps a persistent set of preferred key chunks. Layer-local
Gaussian noise is added to the chunk embeddings. This is a CPU toy, not the
paper's 384 x 192 x 32-layer table.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from routerelay.relay import (
    Ledger,
    build_ledger,
    chunk_scores,
    relay_layer,
    topk_routes,
)


class StackResult(NamedTuple):
    recall: jax.Array
    n_score_pairs: jax.Array
    n_full_pairs: jax.Array
    n_refreshed: jax.Array


def toy_chunk_stack(
    key: jax.Array,
    n_q: int,
    n_k: int,
    d: int,
    n_pref: int,
    drift_std: float,
    n_layers: int,
) -> tuple[jax.Array, jax.Array]:
    """Return Q_c, K_c with shapes [L, n_q, d] and [L, n_k, d].

    Key chunk j lives on axis j, so d must be at least n_k. Preferred keys get
    a large query loading, which keeps a wide top-k margin at low drift.
    """
    if d < n_k:
        raise ValueError("toy_chunk_stack needs d >= n_k so each key owns an axis")
    k1, k2, k3 = jax.random.split(key, 3)
    k_c0 = jnp.zeros((n_k, d))
    k_c0 = k_c0.at[jnp.arange(n_k), jnp.arange(n_k)].set(4.0)
    pref_keys = jax.random.split(k1, n_q)

    def one_pref(pk):
        return jax.random.choice(pk, n_k, shape=(n_pref,), replace=False)

    pref = jax.vmap(one_pref)(pref_keys)
    q_c0 = jnp.zeros((n_q, d))
    rows = jnp.arange(n_q)[:, None]
    q_c0 = q_c0.at[rows, pref].add(4.0)
    q_drift = drift_std * jax.random.normal(k2, (n_layers, n_q, d))
    k_drift = drift_std * jax.random.normal(k3, (n_layers, n_k, d))
    return q_c0[None, :, :] + q_drift, k_c0[None, :, :] + k_drift


def route_recall(routes: jax.Array, exact: jax.Array) -> jax.Array:
    """Mean fraction of exact top-k ids recovered in the relayed route."""
    hits = (exact[:, :, None] == routes[:, None, :]).any(axis=-1)
    return hits.mean()


def run_stack(
    q_c: jax.Array,
    k_c: jax.Array,
    *,
    k: int,
    m: int,
    r: int,
    gamma: float,
    rng_key: jax.Array,
    hysteresis_horizon: int = 1,
) -> StackResult:
    """Run RouteRelay over a precomputed (Q_c, K_c) stack. Layer 0 is an anchor."""
    n_layers, n_q, _ = q_c.shape
    n_k = k_c.shape[1]
    keys = jax.random.split(rng_key, n_layers)
    scores0 = chunk_scores(q_c[0], k_c[0])
    ledger: Ledger = build_ledger(scores0, k, m, r, keys[0])
    hit_sum = route_recall(ledger.routes, topk_routes(scores0, k))
    n_pairs = n_q * n_k
    n_full = n_q * n_k
    n_refreshed = n_q
    for t in range(1, n_layers):
        ledger, info = relay_layer(
            q_c[t],
            k_c[t],
            ledger,
            k=k,
            m=m,
            r=r,
            gamma=gamma,
            rng_key=keys[t],
            hysteresis_horizon=hysteresis_horizon,
        )
        exact = topk_routes(chunk_scores(q_c[t], k_c[t]), k)
        hit_sum = hit_sum + route_recall(ledger.routes, exact)
        n_pairs = n_pairs + int(info.n_score_pairs)
        n_full = n_full + n_q * n_k
        n_refreshed = n_refreshed + int(info.n_refreshed)
    recall = hit_sum / n_layers
    return StackResult(
        recall=jnp.asarray(recall),
        n_score_pairs=jnp.asarray(n_pairs),
        n_full_pairs=jnp.asarray(n_full),
        n_refreshed=jnp.asarray(n_refreshed),
    )
