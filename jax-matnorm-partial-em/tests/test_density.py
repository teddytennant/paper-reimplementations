"""The matrix normal density, its Kronecker convention, and the dtype."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import matnorm_pem as mp
from matnorm_pem import synth


DIMS = [(2, 3), (3, 4), (4, 4), (5, 3)]


def test_x64_is_on():
    """Every downstream tolerance in this suite assumes float64.

    On NVIDIA a float32 matmul defaults to TF32 and loses about 1e-2 relative
    precision, which would quietly pass a loose test and fail a tight one, so
    check the dtype rather than trusting the config flag.
    """
    assert jax.config.jax_enable_x64
    params = synth.paper_params(3, 4)
    Y = mp.sample_matnorm(jr.key(0), params, 4)
    assert Y.dtype == jnp.float64
    assert params.S1.dtype == jnp.float64
    assert mp.matnorm_logpdf(Y[0], params).dtype == jnp.float64
    A = jr.normal(jr.key(1), (64, 64), dtype=jnp.float64)
    err = np.max(np.abs(np.asarray(A @ A.T) - np.asarray(A) @ np.asarray(A).T))
    assert err < 1e-11, "matmul is not running in float64"


@pytest.mark.parametrize("p,q", DIMS)
def test_logpdf_matches_explicit_kron_mvn(p, q):
    """phi_{p x q}(Y) equals N_pq(vec(M), sigma^2 (S2 (x) S1)) of (1)."""
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(p * 10 + q), params, 8)
    Sigma = mp.kron_cov(params.S1, params.S2, params.sigma2)
    for i in range(8):
        a = float(mp.matnorm_logpdf(Y[i], params))
        b = float(mp.mvn_logpdf(mp.vec(Y[i]), mp.vec(params.M), Sigma))
        assert abs(a - b) < 1e-10 * max(1.0, abs(b))


def test_kron_order_is_column_factor_first():
    """Cell (i, j) of Y sits at index j p + i, and the factor order is S2 (x) S1.

    Two checks. The entrywise identity pins the convention, and the average log
    density over many draws pins it statistically: the correct order maximizes
    the expected log density, so swapping the factors has to cost something.
    A single draw is not enough, the two orders can agree by luck on one point.
    """
    p = q = 4
    params = mp.normalize_scale(synth.sinusoidal_mean(p, q),
                                synth.ar1(0.8, p), synth.ar1(0.2, q), 1.2)
    Sigma = np.asarray(mp.kron_cov(params.S1, params.S2, params.sigma2))
    S1 = np.asarray(params.S1)
    S2 = np.asarray(params.S2)
    s2 = float(params.sigma2)
    for i in range(p):
        for j in range(q):
            for k in range(p):
                for l in range(q):
                    assert abs(Sigma[j * p + i, l * p + k] - s2 * S2[j, l] * S1[i, k]) < 1e-12

    Y = mp.sample_matnorm(jr.key(11), params, 2000)
    vY = np.asarray(jax.vmap(mp.vec)(Y))
    mu = mp.vec(params.M)
    right = np.mean([float(mp.mvn_logpdf(v, mu, Sigma)) for v in vY[:400]])
    wrong_cov = s2 * jnp.kron(params.S1, params.S2)
    wrong = np.mean([float(mp.mvn_logpdf(v, mu, wrong_cov)) for v in vY[:400]])
    assert right - wrong > 0.5


def test_sample_covariance_matches_kron():
    """Draws have covariance sigma^2 (S2 (x) S1), not the transposed order."""
    p, q, n = 3, 4, 40000
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(3), params, n)
    X = np.asarray(jax.vmap(mp.vec)(Y - params.M))
    S = X.T @ X / n
    right = np.asarray(mp.kron_cov(params.S1, params.S2, params.sigma2))
    wrong = np.asarray(params.sigma2 * jnp.kron(params.S1, params.S2))
    # 40k draws puts the standard error of each entry near 0.007
    assert np.max(np.abs(S - right)) < 0.06
    assert np.max(np.abs(S - wrong)) > 0.2


@pytest.mark.parametrize("p,q", [(3, 4), (4, 5)])
def test_observed_loglik_reduces_to_the_density(p, q):
    """With nothing missing the Section 4.1 evaluator is just the density."""
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(5), params, 6)
    mask = jnp.zeros((6, p, q), dtype=bool)
    direct = sum(float(mp.matnorm_logpdf(Y[i], params)) for i in range(6))
    assert abs(float(mp.observed_loglik(Y, mask, params)) - direct) < 1e-9
