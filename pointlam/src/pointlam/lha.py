"""Local Hadamard Aggregator (LHA).

Transient-grid neighbor aggregation (specified eq 4):
    f'_i = (1/|N_i|) sum_{j in N_i} (W_v f_j  odot  sigma(W_g [f_i ; Delta p_ij]))
    Delta p_ij = p_j - p_i.  sigma = sigmoid.  [ ; ] is concatenation.

Residual Hadamard gating (specified eq 5):
    Output = LayerNorm( f_in + Linear(f') odot sigma(Linear(f_in)) )
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pointlam.grid import neighbor_mask


class LHAParams(NamedTuple):
    W_v: jax.Array
    W_g: jax.Array
    W1: jax.Array
    W2: jax.Array
    ln_gamma: jax.Array
    ln_beta: jax.Array


def init_lha(key: jax.Array, dim: int) -> LHAParams:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return LHAParams(
        W_v=jax.random.normal(k1, (dim, dim)) * jnp.sqrt(1.0 / dim),
        W_g=jax.random.normal(k2, (dim + 3, dim)) * jnp.sqrt(1.0 / (dim + 3)),
        W1=jax.random.normal(k3, (dim, dim)) * jnp.sqrt(1.0 / dim),
        W2=jax.random.normal(k4, (dim, dim)) * jnp.sqrt(1.0 / dim),
        ln_gamma=jnp.ones((dim,)),
        ln_beta=jnp.zeros((dim,)),
    )


def layer_norm(
    x: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    hat = (x - mean) / jnp.sqrt(var + eps)
    return gamma * hat + beta


def residual_hadamard(
    f_in: jax.Array,
    f_prime: jax.Array,
    W1: jax.Array,
    W2: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Eq 5: LN(f_in + (f' @ W1) odot sigmoid(f_in @ W2))."""
    inner = f_in + (f_prime @ W1) * jax.nn.sigmoid(f_in @ W2)
    return layer_norm(inner, gamma, beta, eps)


def local_hadamard(
    points: jax.Array,
    features: jax.Array,
    mask: jax.Array,
    W_v: jax.Array,
    W_g: jax.Array,
) -> jax.Array:
    """Eq 4. points (N, 3), features (N, D), mask (N, N). Returns f' (N, D)."""
    n, dim = features.shape
    delta = points[None, :, :] - points[:, None, :]
    fi = jnp.broadcast_to(features[:, None, :], (n, n, dim))
    cat = jnp.concatenate([fi, delta], axis=-1)
    gate = jax.nn.sigmoid(cat @ W_g)
    vj = features @ W_v
    term = vj[None, :, :] * gate
    w = mask.astype(features.dtype)[:, :, None]
    k = jnp.maximum(w.sum(axis=1), 1.0)
    return (term * w).sum(axis=1) / k


def apply_lha(
    params: LHAParams,
    points: jax.Array,
    features: jax.Array,
    cell_size: float,
    mask: jax.Array | None = None,
    origin: jax.Array | None = None,
) -> jax.Array:
    if mask is None:
        mask = neighbor_mask(points, cell_size, origin=origin)
    f_prime = local_hadamard(points, features, mask, params.W_v, params.W_g)
    return residual_hadamard(
        features,
        f_prime,
        params.W1,
        params.W2,
        params.ln_gamma,
        params.ln_beta,
    )
