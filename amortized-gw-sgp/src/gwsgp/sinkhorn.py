"""Log-domain Sinkhorn for the Birkhoff polytope (row/column sums = 1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sinkhorn(
    cost: jax.Array,
    epsilon: float = 0.1,
    n_iters: int = 32,
) -> jax.Array:
    """Entropy-regularized projection onto doubly-stochastic matrices.

    Solves ``min_{T in Pi_n} <C, T> - epsilon H(T)`` with uniform marginals
    of ones (permutation-matrix scale, total mass ``n``).

    Parameters
    ----------
    cost:
        Square cost ``C``, shape ``(..., n, n)``.
    epsilon:
        Entropic temperature. Smaller values yield sparser plans.
    n_iters:
        Number of alternating row/column scalings.

    Returns
    -------
    T:
        Coupling of the same shape as ``cost``, with row and column sums
        approximately 1.
    """
    cost = jnp.asarray(cost)
    cost = cost.astype(jnp.result_type(cost, jnp.float32))
    log_k = -cost / epsilon
    n = cost.shape[-1]
    batch_shape = cost.shape[:-2]
    log_u = jnp.zeros(batch_shape + (n,), dtype=cost.dtype)
    log_v = jnp.zeros(batch_shape + (n,), dtype=cost.dtype)

    def body(carry, _):
        log_u, log_v = carry
        log_u = -jax.nn.logsumexp(log_k + log_v[..., None, :], axis=-1)
        log_v = -jax.nn.logsumexp(log_k + log_u[..., :, None], axis=-2)
        return (log_u, log_v), None

    (log_u, log_v), _ = jax.lax.scan(body, (log_u, log_v), None, length=n_iters)
    log_t = log_k + log_u[..., :, None] + log_v[..., None, :]
    return jnp.exp(log_t)
