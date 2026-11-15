"""Velocity MLP with post-linear AdaLN, equation (27) and Appendix C.3.

h = W1 c + b1
h_tilde = GELU((1 + gamma(e_t)) * h + beta(e_t))
v = W2 h_tilde + b2

gamma and beta are one zero-initialised linear projection of a sinusoidal
flow-time embedding e_t. At initialisation, gamma = beta = 0, so
v = W2 GELU(h) + b2.

The PDF does not state the embedding dimension or the frequency schedule.
The default dimension equals the hidden width. Frequencies follow the
standard transformer schedule
omega_i = exp(-log(10000) * i / (dim // 2)), and e_t concatenates
cos(t * omega) and sin(t * omega).
"""

import jax
import jax.numpy as jnp

from mercerflow.common import default_float_dtype


def sinusoidal_time_embedding(t, dim):
    """Sinusoidal embedding of scalar or batched flow time t."""
    t = jnp.asarray(t)
    dim = int(dim)
    if dim < 1:
        raise ValueError("embedding dimension must be positive")
    dtype = t.dtype
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = default_float_dtype()
        t = t.astype(dtype)
    half = dim // 2
    if half == 0:
        return jnp.zeros(t.shape + (dim,), dtype=dtype)
    index = jnp.arange(half, dtype=dtype)
    freqs = jnp.exp(-jnp.log(jnp.asarray(10000.0, dtype=dtype)) * index / half)
    angles = t[..., None] * freqs
    embedding = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    if dim % 2 == 1:
        zeros = jnp.zeros(t.shape + (1,), dtype=dtype)
        embedding = jnp.concatenate([embedding, zeros], axis=-1)
    return embedding


def init_velocity_mlp(key, length, hidden=64, embed_dim=None, dtype=None):
    """Initialise the equation (27) MLP.

    W1 has shape (hidden, L), W2 has shape (L, hidden). W_t has shape
    (2 * hidden, embed_dim) and b_t has shape (2 * hidden,). Both time
    parameters are zero, so the scale and shift start at zero.

    Hidden width defaults to 64, as in Appendix C.3.
    """
    dtype = default_float_dtype() if dtype is None else dtype
    length = int(length)
    hidden = int(hidden)
    if embed_dim is None:
        embed_dim = hidden
    embed_dim = int(embed_dim)
    if length < 1 or hidden < 1 or embed_dim < 1:
        raise ValueError("length, hidden, and embed_dim must be positive")
    key_in, key_out = jax.random.split(key)
    in_scale = jnp.sqrt(jnp.asarray(1.0 / length, dtype=dtype))
    out_scale = jnp.sqrt(jnp.asarray(1.0 / hidden, dtype=dtype))
    W1 = jax.random.normal(key_in, (hidden, length), dtype=dtype) * in_scale
    b1 = jnp.zeros((hidden,), dtype=dtype)
    W2 = jax.random.normal(key_out, (length, hidden), dtype=dtype) * out_scale
    b2 = jnp.zeros((length,), dtype=dtype)
    W_t = jnp.zeros((2 * hidden, embed_dim), dtype=dtype)
    b_t = jnp.zeros((2 * hidden,), dtype=dtype)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W_t": W_t, "b_t": b_t}


def apply_velocity_mlp(params, t, c):
    """Evaluate v_theta(t, c) with AdaLN after the first linear map.

    c has shape (L,) or (batch, L). t is a scalar or shape (batch,).
    """
    c = jnp.asarray(c)
    squeeze = c.ndim == 1
    if squeeze:
        c = c[None, :]
    if c.ndim != 2:
        raise ValueError("c must have shape (L,) or (batch, L)")
    t = jnp.asarray(t, dtype=c.dtype)
    if t.ndim == 0:
        t = jnp.broadcast_to(t, (c.shape[0],))
    elif t.ndim == 1 and t.shape[0] == 1 and c.shape[0] != 1:
        t = jnp.broadcast_to(t, (c.shape[0],))
    elif t.ndim != 1 or t.shape[0] != c.shape[0]:
        raise ValueError("t must be a scalar or match the batch of c")

    h = c @ params["W1"].T + params["b1"]
    embed_dim = params["W_t"].shape[1]
    e_t = sinusoidal_time_embedding(t, embed_dim)
    modulation = e_t @ params["W_t"].T + params["b_t"]
    gamma, beta = jnp.split(modulation, 2, axis=-1)
    h_tilde = jax.nn.gelu((1.0 + gamma) * h + beta, approximate=False)
    v = h_tilde @ params["W2"].T + params["b2"]
    if squeeze:
        return v[0]
    return v


def count_parameters(params):
    """Number of scalar parameters in a pytree of arrays."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(leaf.size) for leaf in leaves))
