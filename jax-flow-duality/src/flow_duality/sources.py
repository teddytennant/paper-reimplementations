"""Continuous source families on R^{..., V} (Haxholli, arXiv:2609.10863)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_gaussian(key, shape):
    """i.i.d. standard normal coordinates, shape ending in V."""
    return jax.random.normal(key, shape)


def sample_bounded_uniform(key, shape, *, width=1.0, low=0.0):
    """i.i.d. uniform coordinates on an interval of length `width`.

    The induced discrete coefficient depends only on the width, not on the
    location of the interval (eq 34).
    """
    width = jnp.asarray(width)
    low = jnp.asarray(low)
    return jax.random.uniform(key, shape, minval=low, maxval=low + width)


def sample_centered_negexp(key, shape, *, beta=1.0):
    """Centered negative-exponential coordinates (eq 35, 56).

    Z = 1/beta - E with E ~ Exp(beta) (rate parameterization), so E[Z] = 0.
    """
    beta = jnp.asarray(beta)
    expo = jax.random.exponential(key, shape) / beta
    return (1.0 / beta) - expo


def sample_source(key, shape, family, *, width=1.0, low=0.0, beta=1.0):
    """Dispatch to a named source family: gaussian, uniform, negexp."""
    family = str(family).lower()
    if family in ("gaussian", "gauss"):
        return sample_gaussian(key, shape)
    if family in ("uniform", "bounded_uniform", "unif"):
        return sample_bounded_uniform(key, shape, width=width, low=low)
    if family in ("negexp", "centered_negexp", "negative_exponential"):
        return sample_centered_negexp(key, shape, beta=beta)
    raise ValueError(f"unknown source family: {family}")
