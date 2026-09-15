"""Algorithm 1 down-up walk on X_k^d (paper Section 2.2 and Section 3).

From x with S = set(x):
  1. Down: T ~ uniform among (k-1)-subsets of S (drop one +1 uniformly).
  2. Up: S' containing T with |S'| = k, sampled proportional to pi(S').

For the SK / Ising measure (eq. (10)), the up-step law of i = S' \\ T is
(Remark 2)
    ∝ exp(β (2 J_ii + 4 ⟨J e_i, 1_T⟩ − 2 ⟨J e_i, 1_d⟩ + 2 h_i)).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from himag.sk import log_unnormalized
from himag.slice import enumerate_slice

Array = jax.Array


def _as_field(h, d: int, dtype) -> Array:
    h = jnp.asarray(h, dtype=dtype)
    if h.ndim == 0:
        return jnp.full((d,), h)
    return h


def up_log_weights(t_mask: Array, J: Array, h, beta: float) -> Array:
    """Unnormalized log-weights for adding each index i to T (Remark 2).

    ``t_mask`` is a length-d boolean (or 0/1) mask of the down-set T.
    """
    t_mask = jnp.asarray(t_mask)
    J = jnp.asarray(J)
    dtype = J.dtype
    ones_t = t_mask.astype(dtype)
    ones_d = jnp.ones((J.shape[0],), dtype=dtype)
    hvec = _as_field(h, J.shape[0], dtype)
    # ⟨J e_i, 1_T⟩ = (J^T 1_T)_i. Model 1 J is symmetric.
    j_t = J.T @ ones_t
    j_all = J.T @ ones_d
    return jnp.asarray(beta, dtype=dtype) * (
        2.0 * jnp.diag(J) + 4.0 * j_t - 2.0 * j_all + 2.0 * hvec
    )


def down_up_step(key: Array, x: Array, J: Array, h, beta: float) -> Array:
    """One step of Algorithm 1 for the SK / Ising measure on the slice of x."""
    x = jnp.asarray(x)
    plus = x > 0
    key_down, key_up = jax.random.split(key)
    drop_idx = jax.random.categorical(key_down, jnp.where(plus, 0.0, -jnp.inf))
    t_mask = plus.at[drop_idx].set(False)
    log_w = up_log_weights(t_mask, J, h, beta)
    add_idx = jax.random.categorical(key_up, jnp.where(~t_mask, log_w, -jnp.inf))
    x_new = jnp.where(
        t_mask,
        jnp.asarray(1.0, dtype=x.dtype),
        jnp.asarray(-1.0, dtype=x.dtype),
    )
    return x_new.at[add_idx].set(jnp.asarray(1.0, dtype=x.dtype))


def transition_matrix(J, h, beta: float, d: int, k: int):
    """Exact Algorithm 1 kernel on X_k^d (tiny d).

    Returns ``(states, P)`` with ``states.shape == (n, d)`` and
    ``P.shape == (n, n)``, where P[i, j] = Prob(x_i → x_j).
    """
    states = np.asarray(enumerate_slice(d, k))
    n = states.shape[0]
    log_pi = np.asarray(log_unnormalized(jnp.asarray(states), J, h, beta))
    pi = np.exp(log_pi - np.max(log_pi))
    plus = states > 0
    sets = [frozenset(np.flatnonzero(plus[s])) for s in range(n)]
    set_to_idx = {S: i for i, S in enumerate(sets)}
    P = np.zeros((n, n), dtype=np.float64)
    inv_k = 1.0 / k
    for s, S in enumerate(sets):
        for drop in S:
            T = S - {drop}
            dests = []
            weights = []
            for j in range(d):
                if j in T:
                    continue
                dest = set_to_idx[T | {j}]
                dests.append(dest)
                weights.append(pi[dest])
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()
            for dest, w in zip(dests, weights):
                P[s, dest] += inv_k * w
    return jnp.asarray(states), jnp.asarray(P)
