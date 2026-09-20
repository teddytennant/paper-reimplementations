"""AIMMD committor MLP, eqs. (3) and (6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

HIDDEN = (20, 10)


def _init_mlp(key, sizes):
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / (n_in + n_out))
        b = jnp.zeros((n_out,))
        params.append((w, b))
    return params


def _mlp(params, x, act=jnp.tanh):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = act(x)
    return x


def init_committor(key, hidden=HIDDEN, in_dim: int = 2):
    sizes = (in_dim,) + tuple(hidden) + (1,)
    return _init_mlp(key, sizes)


def log_committor(params, x):
    """q(x|θ), the log-committor / logit of pB."""
    x = jnp.asarray(x)
    squeeze = x.ndim == 1
    x = jnp.atleast_2d(x)
    qv = _mlp(params, x)[..., 0]
    return qv[0] if squeeze else qv


def q(params, x):
    return log_committor(params, x)


def pB(params, x):
    """p_B(x|θ) = 1 / (1 + exp(-q(x|θ))), eq. (3)."""
    return jax.nn.sigmoid(log_committor(params, x))


def aimmd_loss_from_logits(logits, s):
    """sum_ij log(1 + exp(s_ij q(x_i))), eq. (6). s=-1 if B first, else +1."""
    logits = jnp.asarray(logits)
    s = jnp.asarray(s)
    logits = logits.reshape(-1, 1)
    if s.ndim == 1:
        s = s[:, None]
    return jnp.sum(jnp.logaddexp(0.0, s * logits))


def aimmd_loss(params, x, s):
    return aimmd_loss_from_logits(log_committor(params, x), s)
