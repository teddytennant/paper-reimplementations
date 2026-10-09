"""Forward process, equation 1.

p(z_t | z_0) = Cat(alpha_t * z_0 + (1 - alpha_t) * pi)

Masked kernel: pi is a point mass on the mask id. A non-mask token is kept
with probability alpha_t and becomes mask otherwise.

Uniform kernel: pi = 1/N over the N non-mask tokens (or over the whole
vocabulary if mask_id is None). The generative step keeps the token with
probability alpha_t and otherwise resamples from pi. The categorical
probabilities are the corresponding mixture, so
    P(z_t = z_0) = alpha_t + (1 - alpha_t) * pi[z_0]
    P(z_t = k)   = (1 - alpha_t) * pi[k]    for k != z_0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from refinemix.schedule import EPS, alpha


def mask_prior(vocab_size: int, mask_id: int):
    pi = jnp.zeros((vocab_size,), dtype=jnp.float32)
    return pi.at[mask_id].set(1.0)


def uniform_prior(vocab_size: int, mask_id: int | None = None):
    """pi = 1/N on the N tokens that are not `mask_id` (all tokens if None)."""
    pi = jnp.ones((vocab_size,), dtype=jnp.float32)
    if mask_id is not None:
        pi = pi.at[int(mask_id)].set(0.0)
    return pi / jnp.sum(pi)


def forward_probs(z0, alpha_t, pi):
    """Exact categorical probabilities of equation 1.

    z0: integer array of any shape
    alpha_t: scalar or array broadcastable to z0
    pi: (V,)
    returns: (..., V)
    """
    z0 = jnp.asarray(z0)
    pi = jnp.asarray(pi)
    vocab = pi.shape[0]
    one_hot = jax.nn.one_hot(z0, vocab, dtype=pi.dtype)
    a = jnp.asarray(alpha_t, dtype=pi.dtype)
    while a.ndim < z0.ndim:
        a = a[..., None]
    a = jnp.broadcast_to(a, z0.shape)
    return a[..., None] * one_hot + (1.0 - a[..., None]) * pi


def token_forward_probs(
    z0,
    t,
    *,
    vocab_size: int,
    kernel: str = "masked",
    mask_id: int | None = None,
    eps: float = EPS,
    ideal: bool = False,
):
    """Equation 1 at time t. Returns probability tensor (..., V)."""
    if kernel == "masked":
        if mask_id is None:
            mask_id = vocab_size - 1
        pi = mask_prior(vocab_size, mask_id)
    elif kernel == "uniform":
        pi = uniform_prior(vocab_size, mask_id=mask_id)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    a = alpha(t, eps=eps, ideal=ideal).astype(pi.dtype)
    return forward_probs(z0, a, pi)


def noise_tokens(
    x0,
    t,
    key,
    *,
    kernel: str = "masked",
    mask_id: int | None = None,
    vocab_size: int | None = None,
    eps: float = EPS,
    ideal: bool = False,
):
    """Sample z_t from equation 1. x0 is an integer array (B, L) or (L,)."""
    x0 = jnp.asarray(x0, dtype=jnp.int32)
    a = alpha(t, eps=eps, ideal=ideal)
    a = jnp.asarray(a, dtype=jnp.float32)
    while a.ndim < x0.ndim:
        a = a[..., None]
    a = jnp.broadcast_to(a, x0.shape)
    keep_key, resample_key = jax.random.split(key)
    keep = jax.random.bernoulli(keep_key, a)
    if kernel == "masked":
        if mask_id is None:
            raise ValueError("masked kernel requires mask_id")
        resampled = jnp.full_like(x0, int(mask_id))
    elif kernel == "uniform":
        if vocab_size is None:
            raise ValueError("uniform kernel requires vocab_size")
        if mask_id is None:
            choices = jnp.arange(vocab_size, dtype=jnp.int32)
        else:
            choices = jnp.array(
                [i for i in range(vocab_size) if i != int(mask_id)],
                dtype=jnp.int32,
            )
        idx = jax.random.randint(resample_key, x0.shape, 0, choices.shape[0])
        resampled = choices[idx]
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return jnp.where(keep, x0, resampled)
