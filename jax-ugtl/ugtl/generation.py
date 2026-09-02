"""Candidate generation, Algorithm 6.

Two parts. The sampler in ``ugtl.diffusion`` produces N_gen guided trajectories
anchored on observed prefixes; this module takes their suffixes, screens them
with the proxy mu_phi, and runs Cluster-then-Select to spend the candidate
budget across distinct regions instead of on one mode.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ugtl.diffusion import sample_trajectories
from ugtl.kmeans import kmeans

__all__ = ["target_score", "cluster_then_select", "generate_candidates"]


def target_score(y: np.ndarray, lambda_tar: float) -> float:
    """y_tar = lambda_tar * max_{(x, y) in D} y, the conditioning level of line 3."""
    return float(lambda_tar * np.max(y))


def cluster_then_select(key, pool: np.ndarray, proxy: np.ndarray, Q: int, M_pool: int, n_init: int = 10):
    """Lines 12 to 16 of Algorithm 6.

    Keep the top M_pool designs by mu_phi, partition them into Q clusters with
    k-means, and return the argmax of mu_phi inside each cluster. k-means can
    leave a cluster empty; those slots are filled with the next best unpicked
    design from the pool, so exactly Q distinct designs come back.
    """
    pool = np.asarray(pool, dtype=np.float32)
    proxy = np.asarray(proxy, dtype=np.float32)
    if Q > pool.shape[0]:
        raise ValueError("the candidate budget cannot exceed the pool size")
    keep = np.argsort(-proxy, kind="stable")[: min(M_pool, pool.shape[0])]
    top = pool[keep]

    scale = top.std(axis=0) + 1e-6
    assign, _ = kmeans(key, jnp.asarray((top - top.mean(axis=0)) / scale), Q, n_init=n_init)
    assign = np.asarray(assign)

    chosen: list[int] = []
    for q in range(Q):
        members = np.flatnonzero(assign == q)
        if members.size:
            chosen.append(int(keep[members[np.argmax(proxy[keep][members])]]))
    for j in keep:
        if len(chosen) >= Q:
            break
        if int(j) not in chosen:
            chosen.append(int(j))
    return np.array(chosen[:Q], dtype=np.int64)


def generate_candidates(
    key,
    model,
    trajectories,
    surrogate,
    normalizer,
    y_tar: float,
    Q: int = 16,
    n_gen: int = 256,
    H_ctx: int = 8,
    M_pool: int = 512,
    s_cfg: float = 1.5,
    batch: int = 256,
) -> np.ndarray:
    """Algorithm 6, returning Q designs in the original design space.

    Contexts tau^ctx are drawn from D_traj and their first H_ctx designs anchor
    each generated trajectory (line 4 and Eq. 27). The pool P collects the
    suffixes X_{H_ctx+1:H} of every generated trajectory (line 10). ``normalizer``
    maps between the design space and the [-1, 1] box the diffusion model works
    in, and designs that leave that box are clipped back, which is the validity
    screen of line 12.
    """

    designs = jnp.asarray(trajectories.designs)
    pool = []
    remaining = n_gen
    while remaining > 0:
        m = min(batch, remaining)
        key, k_ctx, k_sample = jax.random.split(key, 3)
        ctx_idx = jax.random.randint(k_ctx, (m,), 0, designs.shape[0])
        ctx = normalizer.forward(designs[ctx_idx, :H_ctx])
        X = sample_trajectories(k_sample, model, m, y_tar, ctx, s_cfg)
        pool.append(np.asarray(normalizer.inverse(X[:, H_ctx:])).reshape(-1, model.dim))
        remaining -= m

    P = np.clip(np.concatenate(pool, axis=0), normalizer.lo, normalizer.hi)
    proxy = np.asarray(surrogate.mean(jnp.asarray(P)))
    key, k_cluster = jax.random.split(key)
    idx = cluster_then_select(k_cluster, P, proxy, Q, M_pool)
    return P[idx]
