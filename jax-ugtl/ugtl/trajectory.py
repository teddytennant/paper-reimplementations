"""Algorithm 5, uncertainty-aware gradient-guided trajectory construction.

The pieces, in the order the paper introduces them:

  * Eq. (22), the score-feasible neighborhood S_k, with the documented fallback
    to the K highest-scoring unvisited designs when it comes out empty;
  * the stabilized cosine alignment a_{k,j} and the temperature kappa_k;
  * Eq. (24), the softmax over S_k that picks the next design;
  * lines 15 and 16, ranking finished trajectories by Delta_y = y_H - y_1 and
    keeping the top ceil(rho N_traj).

Trajectories are advanced in lockstep rather than one at a time. They never
interact, so this changes nothing about the distribution; it just lets one
vmapped call produce the proxy gradient at every current point per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "Trajectories",
    "cosine_alignment",
    "temperature",
    "transition_probs",
    "score_feasible_neighbourhood",
    "construct_trajectories",
    "construct_random_trajectories",
]

C_COS = 1e-8


@dataclass
class Trajectories:
    """A constructed trajectory dataset D_traj.

    designs   (N, H, d)     the design sequence X_tau of each trajectory
    scores    (N, H)        the observed y along it, in whatever units the
                            constructor was handed; ``run_ugtl`` passes scores
                            already scaled to [0, 1]
    indices   (N, H)        which rows of the offline dataset were used
    fallback  (N, H - 1)    True where line 8 of Algorithm 5 had to fire
    """

    designs: np.ndarray
    scores: np.ndarray
    indices: np.ndarray
    fallback: np.ndarray

    @property
    def improvement(self) -> np.ndarray:
        """Delta_y(tau) = y_H - y_1, the ranking key of line 15."""
        return self.scores[:, -1] - self.scores[:, 0]

    def __len__(self) -> int:
        return self.designs.shape[0]


def cosine_alignment(grad: np.ndarray, x_k: np.ndarray, neighbours: np.ndarray, c_cos: float = C_COS) -> np.ndarray:
    """a_{k,j} = <grad mu_phi(x_k), x_j - x_k> / (||grad|| ||x_j - x_k|| + c_cos).

    c_cos = 1e-8 is the paper's value and keeps a zero displacement from dividing
    by zero. Shapes broadcast: grad and x_k are (..., d), neighbours is
    (..., K, d), the result is (..., K).
    """
    delta = neighbours - np.expand_dims(x_k, -2)
    num = np.einsum("...kd,...d->...k", delta, grad)
    den = np.expand_dims(np.linalg.norm(grad, axis=-1), -1) * np.linalg.norm(delta, axis=-1) + c_cos
    return num / den


def temperature(sigma_k, sigma_bar: float, c_temp: float):
    """kappa_k = 1 + sigma_phi(x_k) / (sigma_bar_phi + c_temp).

    Low predictive uncertainty leaves kappa near 1, which sharpens Eq. (24) onto
    the gradient-aligned neighbours. High uncertainty raises kappa and spreads
    the mass over S_k.
    """
    return 1.0 + sigma_k / (sigma_bar + c_temp)


def transition_probs(alignments: np.ndarray, kappa, valid: np.ndarray | None = None) -> np.ndarray:
    """Eq. (24), the softmax of exp(a_{k,j} / kappa_k) normalized over S_k.

    ``valid`` marks which slots really belong to S_k, since a neighborhood can
    come back with fewer than K members.
    """
    kappa = np.expand_dims(np.asarray(kappa), -1) if np.ndim(kappa) else kappa
    logits = np.asarray(alignments) / kappa
    if valid is not None:
        logits = np.where(valid, logits, -np.inf)
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=-1, keepdims=True)


def score_feasible_neighbourhood(dist2, unvisited, y, best, xi, K):
    """Eq. (22) for a batch of trajectory prefixes.

        S_k = NN_K(x_k; {(x_j, y_j) in D : x_j unvisited, y_j >= b_k - xi})

    Returns the K chosen offline indices, a mask saying which slots are real
    members of S_k, and a flag per row for whether line 8 of Algorithm 5 had to
    fire. The fallback keeps the K highest-scoring unvisited designs, breaking
    score ties by distance to x_k.
    """
    feasible = unvisited & (y[None, :] >= (best - xi)[:, None])
    masked = np.where(feasible, dist2, np.inf)
    # A full stable argsort rather than argpartition: equal distances then go to
    # the lower index, which is what jax.lax.top_k does in _nearest_feasible.
    # argpartition leaves ties in whatever order it happened to produce.
    picked = np.argsort(masked, axis=1, kind="stable")[:, :K]
    valid = np.isfinite(np.take_along_axis(masked, picked, axis=1))
    fallback = ~feasible.any(axis=1)
    for u in np.flatnonzero(fallback):
        cand = np.flatnonzero(unvisited[u])
        chosen = cand[np.lexsort((dist2[u, cand], -y[cand]))[:K]]
        picked[u], valid[u] = 0, False
        picked[u, : len(chosen)] = chosen
        valid[u, : len(chosen)] = True
    return picked, valid, fallback


def _start_indices(rng, y, p_init, n_traj):
    """Line 4: sample the first point from the bottom p_init percent of scores."""
    cutoff = np.percentile(y, p_init)
    pool = np.flatnonzero(y <= cutoff)
    return rng.choice(pool, size=n_traj, replace=len(pool) < n_traj)


@partial(jax.jit, static_argnums=(5,))
def _nearest_feasible(X, y, cur, visited, best, K, xi):
    """The device version of Eq. (22): the K nearest unvisited designs scoring at
    least b_k - xi. ``score_feasible_neighbourhood`` is the readable reference
    and the tests hold the two to the same answer.

    The distances are the plain squared differences, not the ||a||^2 + ||b||^2 -
    2 a.b identity. The identity is cheaper in memory, (m, N) instead of
    (m, N, d), but it routes the distance through a float32 matmul, and XLA runs
    those on tensor cores at TF32 by default on an NVIDIA GPU. TF32 keeps 10
    mantissa bits, so on the Branin data, where ||x||^2 reaches ~300, the
    cancellation left errors up to 0.2 in a d2 whose 10th and 11th nearest
    neighbours are 0.019 apart. That reorders the neighborhood. The subtraction
    below is elementwise, never a matmul, so nothing downgrades it.
    """
    delta = X[cur][:, None, :] - X[None, :, :]
    d2 = (delta * delta).sum(axis=-1)
    feasible = (~visited) & (y[None, :] >= (best - xi)[:, None])
    masked = jnp.where(feasible, d2, jnp.inf)
    # top_k breaks ties by lowest index; score_feasible_neighbourhood sorts
    # stably so it does the same.
    neg, picked = jax.lax.top_k(-masked, K)
    return picked, jnp.isfinite(neg), feasible.any(axis=1)


@jax.jit
def _sample_next(key, X, cur, picked, valid, grads, kappa):
    """Eq. (24) on device. Same logits as ``transition_probs``, sampled with
    ``jax.random.categorical``."""
    delta = X[picked] - X[cur][:, None, :]
    # Same reason as _nearest_feasible: an einsum lowers to a dot, and a float32
    # dot is TF32 on GPU. delta and grads are small, so broadcast and sum.
    num = (delta * grads[:, None, :]).sum(axis=-1)
    den = jnp.linalg.norm(grads, axis=-1)[:, None] * jnp.linalg.norm(delta, axis=-1) + C_COS
    logits = jnp.where(valid, (num / den) / kappa[:, None], -jnp.inf)
    slot = jax.random.categorical(key, logits)
    return jnp.take_along_axis(picked, slot[:, None], axis=1)[:, 0]


def construct_trajectories(
    key,
    X: np.ndarray,
    y: np.ndarray,
    surrogate,
    horizon: int = 32,
    n_traj: int = 256,
    p_init: float = 20.0,
    K: int = 20,
    xi: float = 0.05,
    rho: float = 0.5,
    c_temp: float = 0.5,
    chunk: int = 512,
) -> Trajectories:
    """Algorithm 5 end to end.

    The arguments mirror the algorithm box: horizon H, trajectory count N_traj,
    initial percentile p_init, neighborhood size K, score tolerance xi, retention
    ratio rho, and the uncertainty temperature constant c_temp. ``chunk`` only
    bounds how many trajectories share one distance matrix.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    N = X.shape[0]
    if horizon > N:
        raise ValueError("the horizon cannot exceed the number of offline designs")
    if K >= N:
        raise ValueError("the neighborhood size must be smaller than the dataset")
    key, k_start = jax.random.split(key)
    rng = np.random.default_rng(int(jax.random.randint(k_start, (), 0, 2**30)))

    sigma_all = np.asarray(surrogate.dispersion(X))
    sigma_bar = float(sigma_all.mean())
    kappa_all = jnp.asarray(temperature(sigma_all, sigma_bar, c_temp))
    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    idx = np.zeros((n_traj, horizon), dtype=np.int64)
    fallback = np.zeros((n_traj, horizon - 1), dtype=bool)
    idx[:, 0] = _start_indices(rng, y, p_init, n_traj)
    visited = np.zeros((n_traj, N), dtype=bool)
    visited[np.arange(n_traj), idx[:, 0]] = True
    best = y[idx[:, 0]].copy()  # b_k of Eq. (22)

    grad_fn = jax.jit(surrogate.proxy_gradient)
    for k in range(horizon - 1):
        for lo in range(0, n_traj, chunk):
            hi = min(lo + chunk, n_traj)
            key, k_step = jax.random.split(key)
            cur = jnp.asarray(idx[lo:hi, k])
            picked, valid, has_feasible = _nearest_feasible(
                Xj, yj, cur, jnp.asarray(visited[lo:hi]), jnp.asarray(best[lo:hi]), K, xi
            )
            need_fallback = ~np.asarray(has_feasible)
            if need_fallback.any():
                picked, valid = np.asarray(picked).copy(), np.asarray(valid).copy()
                for u in np.flatnonzero(need_fallback):
                    picked[u], valid[u] = _fallback(X, y, visited[lo + u], idx[lo + u, k], K)
                picked, valid = jnp.asarray(picked), jnp.asarray(valid)

            chosen = np.asarray(
                _sample_next(k_step, Xj, cur, picked, valid, grad_fn(Xj[cur]), kappa_all[cur])
            )
            fallback[lo:hi, k] = need_fallback
            idx[lo:hi, k + 1] = chosen
            visited[np.arange(lo, hi), chosen] = True
            best[lo:hi] = np.maximum(best[lo:hi], y[chosen])

    return rank_and_retain(idx, X, y, fallback, rho)


def _fallback(X, y, visited_row, cur, K):
    """Line 8: the K highest-scoring unvisited designs, ties broken by distance."""
    cand = np.flatnonzero(~visited_row)
    d2 = ((X[cand] - X[cur]) ** 2).sum(axis=1)
    chosen = cand[np.lexsort((d2, -y[cand]))[:K]]
    picked = np.zeros(K, dtype=np.int32)
    valid = np.zeros(K, dtype=bool)
    picked[: len(chosen)] = chosen
    valid[: len(chosen)] = True
    return picked, valid


def construct_random_trajectories(
    key,
    X: np.ndarray,
    y: np.ndarray,
    horizon: int = 32,
    n_traj: int = 256,
    p_init: float = 20.0,
    rho: float = 0.5,
) -> Trajectories:
    """The ablation constructor: same start rule and same retention rule, but
    each step jumps to a uniformly random unvisited design.

    Section 4.2 argues that paths assembled without locality do not describe how
    an optimizer moves. This is that constructor, and it is what the end-to-end
    test compares UGTL against.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    N = X.shape[0]
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
    idx = np.zeros((n_traj, horizon), dtype=np.int64)
    idx[:, 0] = _start_indices(rng, y, p_init, n_traj)
    for u in range(n_traj):
        seen = {int(idx[u, 0])}
        for k in range(horizon - 1):
            j = int(rng.integers(N))
            while j in seen:
                j = int(rng.integers(N))
            seen.add(j)
            idx[u, k + 1] = j
    return rank_and_retain(idx, X, y, np.zeros((n_traj, horizon - 1), dtype=bool), rho)


def rank_and_retain(idx, X, y, fallback, rho) -> Trajectories:
    """Lines 15 and 16: rank by Delta_y = y_H - y_1, keep the top ceil(rho N_traj)."""
    scores = y[idx]
    delta = scores[:, -1] - scores[:, 0]
    n_keep = int(np.ceil(rho * idx.shape[0]))
    keep = np.argsort(-delta, kind="stable")[:n_keep]
    return Trajectories(X[idx[keep]], scores[keep], idx[keep], fallback[keep])
