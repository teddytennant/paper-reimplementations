"""Lemma 3.2: i.i.d. permutation-invariant sources have uniform argmax."""

import jax
import jax.numpy as jnp

from flow_duality.interpolant import argmax_tokens
from flow_duality.sources import (
    sample_bounded_uniform,
    sample_centered_negexp,
    sample_gaussian,
)


def _check_uniform_argmax(tokens, vocab_size, tol=0.02):
    n = tokens.shape[0]
    counts = jnp.bincount(tokens, length=vocab_size)
    p = counts / n
    assert jnp.all(jnp.abs(p - 1.0 / vocab_size) < tol), p


def test_lemma_32_uniform_argmax():
    v = 5
    n = 20000
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    g = sample_gaussian(k1, (n, v))
    u = sample_bounded_uniform(k2, (n, v), width=1.0, low=-0.3)
    e = sample_centered_negexp(k3, (n, v), beta=1.0)
    _check_uniform_argmax(argmax_tokens(g), v)
    _check_uniform_argmax(argmax_tokens(u), v)
    _check_uniform_argmax(argmax_tokens(e), v)
