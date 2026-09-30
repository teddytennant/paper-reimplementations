"""CPU checks for the closed-form Gride kernel. Fixed keys, no network."""

import math

import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import quad
from scipy.special import beta as beta_fn
from scipy.special import digamma as scipy_digamma

from gride_id import (
    estimate_id,
    gride_kl,
    gride_mle,
    i_gamma,
    pairwise_ratios,
    von_mises_kl,
)


def _quadrature_i(gamma, k1, k2):
    """Quadrature of I(gamma) via the S1.6 split.

    I(gamma) = gamma * E[-log T] + E[log(1 - T**gamma)] for
    T ~ Beta(k1, k2-k1). E[-log T] = digamma(k2) - digamma(k1).
    The finite sum in S1.9 equals this full expectation, not the
    E[log(1 - T**gamma)] term by itself.
    """
    a = k1
    b = k2 - k1
    partition = beta_fn(a, b)

    def integrand(t):
        if t <= 0.0 or t >= 1.0:
            return 0.0
        gap = 1.0 - t**gamma
        if gap <= 0.0:
            return 0.0
        density = (t ** (a - 1.0)) * ((1.0 - t) ** (b - 1.0)) / partition
        return density * math.log(gap)

    expectation, err = quad(integrand, 0.0, 1.0, epsabs=1e-12, limit=1000)
    full = expectation + gamma * (scipy_digamma(k2) - scipy_digamma(k1))
    return full, err


@pytest.mark.parametrize(
    "d,k1,k2",
    [
        (4.0, 1, 2),
        (7.0, 5, 10),
        (2.5, 3, 8),
        (1.0, 2, 4),
        (10.0, 5, 12),
    ],
)
def test_gride_kl_zero_on_diagonal(d, k1, k2):
    assert abs(gride_kl(d, d, k1, k2)) < 1e-6


@pytest.mark.parametrize(
    "k1,k2,gamma",
    [(1, 2, 1.0), (5, 10, 0.5), (5, 10, 2.0), (3, 8, 1.0)],
)
def test_i_gamma_matches_quadrature(k1, k2, gamma):
    quad_val, err = _quadrature_i(gamma, k1, k2)
    assert err < 1e-8
    assert abs(i_gamma(gamma, k1, k2) - quad_val) < 1e-5


def test_two_nn_mle_matches_closed_form():
    n = 2000
    d_true = 4.0
    u = jax.random.uniform(jax.random.PRNGKey(1), (n,), dtype=jnp.float64)
    u = jnp.clip(u, 0.0, 1.0 - 1e-12)
    mus = (1.0 - u) ** (-1.0 / d_true)
    estimate = gride_mle(mus, 1, 2)
    closed = n / float(jnp.sum(jnp.log(mus)))
    assert abs(estimate - d_true) < 0.15
    assert abs(estimate - closed) < 1e-2


def test_ball_distance_mle_below_ambient():
    key_dir, key_rad = jax.random.split(jax.random.PRNGKey(0))
    gauss = jax.random.normal(key_dir, (400, 3), dtype=jnp.float64)
    gauss = gauss / jnp.linalg.norm(gauss, axis=1, keepdims=True)
    radius = jax.random.uniform(key_rad, (400, 1), dtype=jnp.float64) ** (1.0 / 3.0)
    cloud = jnp.pad(gauss * radius, ((0, 0), (0, 5)))
    assert cloud.shape == (400, 8)
    mus = pairwise_ratios(cloud, 5, 10)
    estimate = gride_mle(mus, 5, 10)
    assert 2.0 <= estimate <= 4.5


def test_profiled_kl_ignores_location_shift():
    nu1, tau1 = 0.2, 1.5
    nu2, tau2 = 0.9, 2.2
    aligned = von_mises_kl(nu1, tau1, nu2, tau2, profiled=True)
    shifted = von_mises_kl(nu1, tau1, nu2 + 1.3, tau2, profiled=True)
    assert abs(aligned - shifted) < 1e-10

    tau = 1.2
    nu_a = 0.4
    full = von_mises_kl(nu_a, tau, nu_a + 1.0, tau, profiled=False)
    profiled = von_mises_kl(nu_a, tau, nu_a + 1.0, tau, profiled=True)
    assert full > profiled


def test_estimate_id_embedded_gaussian():
    key_data, key_est = jax.random.split(jax.random.PRNGKey(0))
    latent = jax.random.normal(key_data, (300, 2), dtype=jnp.float64)
    cloud = jnp.pad(latent, ((0, 0), (0, 4)))
    out = estimate_id(
        cloud,
        k=5,
        m_max=6,
        n_ref=128,
        key=key_est,
        profiled=False,
    )
    assert out["d_hat"] in {1, 2, 3, 4}
    assert out["j"].shape == (6,)
    assert isinstance(out["d_hat"], int)


def test_too_few_points_raises():
    cloud = jnp.zeros((8, 4), dtype=jnp.float64)
    with pytest.raises(ValueError):
        estimate_id(cloud, k=5, key=jax.random.PRNGKey(0))
