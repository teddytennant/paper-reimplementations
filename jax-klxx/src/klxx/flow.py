"""RealNVP-style affine coupling flow G: target -> source, plus source/target potentials."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def source_potential(x):
    """U0(x) = 0.5 ||x||^2, standard Gaussian up to an additive constant."""
    return 0.5 * jnp.sum(x * x, axis=-1)


def mixture_potential(y, means, log_weights, sigma):
    """U(y) = -log sum_k w_k N(y; mu_k, sigma^2 I), stable logsumexp.

    y is a single point of shape (d,).
    """
    d = y.shape[-1]
    diffs = y - means
    log_norm = -0.5 * d * jnp.log(2.0 * jnp.pi * sigma**2)
    log_n = log_norm - jnp.sum(diffs * diffs, axis=-1) / (2.0 * sigma**2)
    return -jax.nn.logsumexp(log_weights + log_n)


def two_mode_potential(y, sigma=0.3):
    """Equal-weight 2D two-mode GMM at (+/-2, 0) with variance sigma^2 I."""
    means = jnp.array([[-2.0, 0.0], [2.0, 0.0]])
    log_weights = jnp.log(jnp.array([0.5, 0.5]))
    return mixture_potential(y, means, log_weights, sigma)


def init_coupling(key, dim=2, n_layers=6, hidden=32):
    """Near-identity affine coupling: last linear layer starts at 0 so G(y)=y."""
    params = []
    for _ in range(n_layers):
        key, k1 = jax.random.split(key)
        w1 = 0.1 * jax.random.normal(k1, (dim, hidden))
        b1 = jnp.zeros((hidden,))
        # Zero last layer: s = t = 0, so each coupling is the identity.
        w2 = jnp.zeros((hidden, 2 * dim))
        b2 = jnp.zeros((2 * dim,))
        params.append({"w1": w1, "b1": b1, "w2": w2, "b2": b2})
    return params


def _conditioner(layer, y_id):
    h = jnp.tanh(y_id @ layer["w1"] + layer["b1"])
    st = h @ layer["w2"] + layer["b2"]
    d = y_id.shape[-1]
    s_raw, t_raw = st[:d], st[d:]
    return s_raw, t_raw


def _forward_one(params, y):
    dim = y.shape[-1]
    logdet = jnp.zeros((), dtype=y.dtype)
    x = y
    for i, layer in enumerate(params):
        mask = (jnp.arange(dim) % 2).astype(y.dtype)
        if i % 2 == 1:
            mask = 1.0 - mask
        s_raw, t_raw = _conditioner(layer, x * mask)
        s = jnp.tanh(s_raw) * (1.0 - mask)
        t = t_raw * (1.0 - mask)
        x = (x - t) * jnp.exp(-s)
        logdet = logdet - jnp.sum(s)
    return x, logdet


def _inverse_one(params, x):
    dim = x.shape[-1]
    logdet_inv = jnp.zeros((), dtype=x.dtype)
    y = x
    for i, layer in reversed(list(enumerate(params))):
        mask = (jnp.arange(dim) % 2).astype(x.dtype)
        if i % 2 == 1:
            mask = 1.0 - mask
        s_raw, t_raw = _conditioner(layer, y * mask)
        s = jnp.tanh(s_raw) * (1.0 - mask)
        t = t_raw * (1.0 - mask)
        y = y * jnp.exp(s) + t
        logdet_inv = logdet_inv + jnp.sum(s)
    return y, logdet_inv


def forward(params, y):
    """G: target y -> source x, plus log|det JG(y)|."""
    if y.ndim == 1:
        return _forward_one(params, y)
    return jax.vmap(lambda yy: _forward_one(params, yy))(y)


def inverse(params, x):
    """G^{-1}: source x -> target y, plus log|det JG^{-1}(x)|."""
    if x.ndim == 1:
        return _inverse_one(params, x)
    return jax.vmap(lambda xx: _inverse_one(params, xx))(x)
