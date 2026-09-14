"""Closed-form k_t versus Monte Carlo, plus algebraic identities."""

import jax
import jax.numpy as jnp

from flow_duality.coefficients import k_gaussian, k_negexp, k_uniform
from flow_duality.duality import induced_coefficient_mc


def test_negexp_closed_form_vs_mc():
    key = jax.random.PRNGKey(1)
    times = (0.2, 0.5, 0.8)
    for v in (2, 16):
        for t in times:
            key, sk = jax.random.split(key)
            closed = float(k_negexp(t, beta=1.0, vocab_size=v))
            mc = float(
                induced_coefficient_mc(
                    sk, t, v, family="negexp", n_samples=8000, beta=1.0
                )
            )
            assert abs(closed - mc) < 0.03, (v, t, closed, mc)


def test_uniform_closed_form_vs_mc():
    key = jax.random.PRNGKey(2)
    v = 4
    eps = 1.0
    for t in (0.2, 0.5, 0.8):
        key, sk = jax.random.split(key)
        closed = float(k_uniform(t, v, eps=eps))
        mc = float(
            induced_coefficient_mc(
                sk, t, v, family="uniform", n_samples=8000, width=eps
            )
        )
        assert abs(closed - mc) < 0.03, (t, closed, mc)


def test_gaussian_closed_form_vs_mc():
    key = jax.random.PRNGKey(3)
    v = 4
    for t in (0.3, 0.6):
        key, sk = jax.random.split(key)
        closed = float(k_gaussian(t, v))
        mc = float(
            induced_coefficient_mc(sk, t, v, family="gaussian", n_samples=8000)
        )
        assert abs(closed - mc) < 0.04, (t, closed, mc)


def test_algebra_identities():
    v = 6
    # r = 1 when ct/eps >= 1; with eps=1 this holds at k_tilde=1 and at t>=0.5.
    assert abs(float(k_uniform(1.0, v, eps=1.0)) - 1.0) < 1e-5
    r_one = float(k_uniform(0.5, v, eps=1.0))
    assert abs(r_one - 1.0) < 1e-5

    t = 0.4
    a = float(k_negexp(t, beta=1.0, vocab_size=2))
    b = float(k_negexp(t, beta=1.0, vocab_size=32))
    assert abs(a - b) < 1e-6

    for fn in (
        lambda s: k_gaussian(s, v),
        lambda s: k_uniform(s, v, eps=1.0),
        lambda s: k_negexp(s, beta=1.0, vocab_size=v),
    ):
        assert abs(float(fn(0.0))) < 1e-5
        assert abs(float(fn(1.0)) - 1.0) < 1e-5
