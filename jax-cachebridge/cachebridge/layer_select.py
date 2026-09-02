"""Source-layer selection, first paragraph of section 3.1.

Centered single-source-layer OLS probes under the head assignment a(h), held-in R^2
averaged across KV heads and over K and V, layers ranked, first k indices give S_l.
CacheBridge inherits this selector from FULL-HEAD MAPPING unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _probe_r2(x, y, ridge: float):
    """Held-in R^2 of a centered OLS probe from x to y, pooled over output coordinates."""
    xc = x - x.mean(0, keepdims=True)
    yc = y - y.mean(0, keepdims=True)
    gram = xc.T @ xc
    w = jnp.linalg.solve(gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype), xc.T @ yc)
    ss_res = jnp.sum((yc - xc @ w) ** 2)
    ss_tot = jnp.sum(yc**2)
    return 1.0 - ss_res / jnp.maximum(ss_tot, 1e-30)


def probe_scores(src_k, src_v, tgt_k, tgt_v, assign, ridge: float = 1e-6):
    """Per (target layer, source layer) probe score, shape (Lt, Ls).

    src_* are (Ls, n, Hs, ds) and tgt_* are (Lt, n, Ht, dt), both content space.
    """
    def per_pair(tk, tv, sk, sv):
        def per_head(h):
            g = assign[h]
            return 0.5 * (_probe_r2(sk[:, g, :], tk[:, h, :], ridge)
                          + _probe_r2(sv[:, g, :], tv[:, h, :], ridge))

        return jnp.mean(jax.vmap(per_head)(jnp.arange(tgt_k.shape[2])))

    over_src = jax.vmap(per_pair, in_axes=(None, None, 0, 0))
    return jax.vmap(over_src, in_axes=(0, 0, None, None))(tgt_k, tgt_v, src_k, src_v)


def select_layers(src_k, src_v, tgt_k, tgt_v, assign, k: int, ridge: float = 1e-6):
    """S_l for every target layer: the top-k source layers by probe score, shape (Lt, k).

    Indices are returned in ascending layer order so the gather schedule is contiguous.
    """
    scores = probe_scores(src_k, src_v, tgt_k, tgt_v, assign, ridge)
    top = jnp.argsort(-scores, axis=1)[:, :k]
    return jnp.sort(top, axis=1), scores
