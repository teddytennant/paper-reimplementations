"""The closed-form one-step denoiser of Sec. II-C, and the attention forms it takes.

Everything here follows from one observation in the paper: if you fix the noisy
input y and average the DDPM loss (6) over which dataset point could have
produced y, you get a convex QP whose minimizer is a softmax-weighted average of
the dataset. That minimizer is eq (11), and under a mild condition on the data
norms it is literally dot-product attention, eq (12).

Shapes: `data` is (N, d), a query `y` is (..., d), and the returned denoised
signal has the same shape as `y`.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "squared_distances",
    "gaussian_likelihood",
    "attention_weights",
    "optimal_denoiser",
    "qp_objective",
    "dot_product_attention",
    "dot_product_attention_weights",
    "softmax_as_attention",
    "one_hot_dataset",
]


def _softmax(logits):
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    e = jnp.exp(shifted)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def squared_distances(y, data):
    """||y - s_i||^2 for every s_i in the dataset. Returns (..., N).

    Expanded as ||y||^2 - 2<y, s_i> + ||s_i||^2, which is the same split eq (12)
    is built on. Writing it this way keeps a batch of queries at (batch, N)
    instead of materializing (batch, N, d), which matters once the dataset is
    large enough to be worth a GPU. The clamp at zero mops up rounding on points
    that coincide.
    """
    y = jnp.asarray(y)
    data = jnp.asarray(data)
    y2 = jnp.sum(y * y, axis=-1)[..., None]
    s2 = jnp.sum(data * data, axis=-1)
    return jnp.maximum(y2 - 2.0 * (y @ data.T) + s2, 0.0)


def gaussian_likelihood(y, data, sigma):
    """p(y | s_i) of eq (7), with the paper's scalar normalization 1/sqrt(2 pi sigma^2).

    The constant is the same for every i, so it cancels in eq (11) and does not
    move the argmin of eq (7). It is here so the loss can be written as the paper
    writes it.
    """
    sigma = jnp.asarray(sigma)
    d2 = squared_distances(y, data)
    norm = 1.0 / jnp.sqrt(2.0 * jnp.pi * sigma**2)
    return norm * jnp.exp(-d2 / (2.0 * sigma**2))


def attention_weights(y, data, sigma):
    """The normalized coefficients p_i / sum_j p_j of eq (11). Returns (..., N).

    Computed as a softmax of -||y - s_i||^2 / (2 sigma^2) so that the shared
    exp underflow is subtracted off rather than divided out.
    """
    sigma = jnp.asarray(sigma)
    logits = -squared_distances(y, data) / (2.0 * sigma**2)
    return _softmax(logits)


def optimal_denoiser(y, data, sigma):
    """Eq (11): z*(y) = sum_i p_i s_i / sum_i p_i, p_i = exp(-||y - s_i||^2 / 2 sigma^2).

    The minimizer of the averaged DDPM loss (7), which the paper rewrites as the
    QP (8)-(9) and solves with the gradient condition (10). A converged denoiser
    trained on the finite dataset D can do no better than this at any y.
    """
    w = attention_weights(y, data, sigma)
    return w @ jnp.asarray(data)


def qp_objective(z, y, data, sigma, normalized=True):
    """The per-y averaged loss, eq (7) written as the QP objective of eq (8).

    sum_i p_i ||z - s_i||^2, with p_i either the full Gaussian likelihood of
    eq (7) (`normalized=True`) or the bare exponential of eq (11). Both have the
    same minimizer, eq (11).
    """
    z = jnp.asarray(z)
    data = jnp.asarray(data)
    if normalized:
        p = gaussian_likelihood(y, data, sigma)
    else:
        p = jnp.exp(-squared_distances(y, data) / (2.0 * jnp.asarray(sigma) ** 2))
    resid = z[..., None, :] - data
    return jnp.sum(p * jnp.sum(resid * resid, axis=-1), axis=-1)


def dot_product_attention_weights(y, data, sigma):
    """Weights of eq (12): softmax over i of <y, s_i> / sigma^2. Returns (..., N)."""
    y = jnp.asarray(y)
    data = jnp.asarray(data)
    sigma = jnp.asarray(sigma)
    return _softmax((y @ data.T) / sigma**2)


def dot_product_attention(y, data, sigma):
    """Eq (12): Attention{y, {s_i}}, the dot-product form of eq (11).

    Expanding ||y - s_i||^2 = ||y||^2 - 2<y, s_i> + ||s_i||^2 in eq (11) kills the
    ||y||^2 term, since it is shared by every i and cancels in the normalization.
    What is left is exact dot-product attention times exp(-||s_i||^2 / 2 sigma^2).
    So eq (12) equals eq (11) exactly when the ||s_i|| are constant, and the gap
    grows with their spread. See `tests/test_attention_equivalence.py`.

    The paper notes that conventional attention takes sigma^2 = sqrt(p) for input
    dimension p, which is the usual 1/sqrt(d_k) scaling.
    """
    w = dot_product_attention_weights(y, data, sigma)
    return w @ jnp.asarray(data)


def one_hot_dataset(n, dtype=None):
    """The dataset {I_i} of standard one-hot label vectors used in eqs (13)-(15)."""
    return jnp.eye(n, dtype=dtype)


def softmax_as_attention(y):
    """Eqs (13)-(15): Softmax{y} = Attention{y, {I_i}}.

    Runs the closed form (11) with sigma = 1 on the one-hot dataset. Then
    p_i = exp(-(||y||^2 - 2 y_i + 1) / 2) is proportional to exp(y_i), so the
    convex combination of one-hot vectors is the softmax vector itself.
    """
    y = jnp.asarray(y)
    return optimal_denoiser(y, one_hot_dataset(y.shape[-1], dtype=y.dtype), 1.0)
