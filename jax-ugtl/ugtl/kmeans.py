"""k-means with k-means++ seeding, for the clustering step of Algorithm 6.

The paper runs k-means++ with 10 independent starts and keeps the best. Same
here, with the best start chosen by inertia. Both the seeding and Lloyd's
iterations run under a single jit, so adding a centre does not reshape anything
and nothing recompiles per cluster.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["kmeans", "kmeans_plusplus"]


@partial(jax.jit, static_argnums=(2,))
def kmeans_plusplus(key, X: jnp.ndarray, k: int) -> jnp.ndarray:
    """Pick k initial centres, each drawn with probability proportional to its
    squared distance to the nearest centre already chosen."""
    n = X.shape[0]
    k0, k_rest = jax.random.split(key)
    first = jax.random.randint(k0, (), 0, n)
    centres = jnp.zeros((k, X.shape[1]), dtype=X.dtype).at[0].set(X[first])
    d2 = ((X - X[first]) ** 2).sum(axis=-1)

    def body(carry, step):
        centres, d2 = carry
        i, sub = step
        p = d2 / jnp.maximum(d2.sum(), 1e-12)
        idx = jax.random.choice(sub, n, p=p)
        centres = centres.at[i].set(X[idx])
        return (centres, jnp.minimum(d2, ((X - X[idx]) ** 2).sum(axis=-1))), None

    (centres, _), _ = jax.lax.scan(
        body, (centres, d2), (jnp.arange(1, k), jax.random.split(k_rest, k - 1))
    )
    return centres


@partial(jax.jit, static_argnums=(2, 3, 4))
def _fit(key, X, k, iters, n_init):
    def one_start(sub):
        centres = kmeans_plusplus(sub, X, k)

        def step(centres, _):
            assign = jnp.argmin(((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1)
            onehot = jax.nn.one_hot(assign, k)
            counts = onehot.sum(0)[:, None]
            return jnp.where(counts > 0, (onehot.T @ X) / jnp.maximum(counts, 1.0), centres), None

        centres, _ = jax.lax.scan(step, centres, None, length=iters)
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        return centres, jnp.argmin(d2, axis=1), jnp.min(d2, axis=1).sum()

    centres, assign, inertia = jax.vmap(one_start)(jax.random.split(key, n_init))
    best = jnp.argmin(inertia)
    return assign[best], centres[best]


def kmeans(key, X: jnp.ndarray, k: int, iters: int = 25, n_init: int = 10):
    """Return (assignments, centres) from the best of ``n_init`` restarts."""
    X = jnp.asarray(X, dtype=jnp.float32)
    if k >= X.shape[0]:
        return jnp.arange(X.shape[0]), X
    return _fit(key, X, k, iters, n_init)
