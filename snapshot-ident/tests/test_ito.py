"""Ito identity and trapezoidal O(Δ^3) residual on a Gaussian OU (Ω = 0)."""

import jax.numpy as jnp

from snapshot_ident.moments import (
    e_generator_coord,
    e_generator_sqnorm,
    e_psi_coord,
    e_psi_sqnorm,
    ito_coord_pair,
    ito_sqnorm_pair,
    trapezoidal_residual,
)
from snapshot_ident.sde import ou_covariance, ou_mean, planted_curl_H, snapshot_covariances


D = 4
NU = 0.7
SIGMA2 = 0.5
TIMES = jnp.array([0.0, 0.35, 0.70, 1.05], dtype=jnp.float64)
SIGMA0 = jnp.diag(jnp.array([2.4, 1.1, 0.8, 0.5], dtype=jnp.float64))
MU0 = jnp.array([0.9, -0.4, 0.2, 0.0], dtype=jnp.float64)


def _ou_path(omega: float = 0.0):
    H = planted_curl_H(D, m=2, nu=NU, omega=omega)
    Omega = H + NU * jnp.eye(D)
    Sigmas = snapshot_covariances(SIGMA0, TIMES, NU, SIGMA2, Omega)
    mus = jax_vmap_mean(H)
    return H, Sigmas, mus


def jax_vmap_mean(H):
    return jnp.stack([ou_mean(MU0, t, H) for t in TIMES])


def test_ito_identity_sqnorm():
    H, Sigmas, mus = _ou_path(omega=0.0)
    for k in range(TIMES.shape[0]):
        ddt, gen = ito_sqnorm_pair(H, Sigmas[k], SIGMA2, mu=mus[k])
        assert jnp.allclose(ddt, gen, rtol=1e-10, atol=1e-12)
        gen2 = e_generator_sqnorm(H, Sigmas[k], SIGMA2, mu=mus[k])
        assert jnp.allclose(gen, gen2, rtol=1e-12, atol=1e-14)


def test_ito_identity_linear_coord():
    H, _, mus = _ou_path(omega=0.0)
    for k in range(TIMES.shape[0]):
        ddt, gen = ito_coord_pair(H, mus[k], i=0)
        assert jnp.allclose(ddt, gen, rtol=1e-12, atol=1e-14)
        assert jnp.allclose(gen, e_generator_coord(H, mus[k], i=0))
        assert jnp.allclose(e_psi_coord(mus[k], i=0), mus[k, 0])


def test_trapezoidal_residual_is_o_delta_cubed():
    H = planted_curl_H(D, m=2, nu=NU, omega=0.0)
    Omega = jnp.zeros((D, D), dtype=jnp.float64)
    t0 = 0.15
    deltas = jnp.array([0.40, 0.20, 0.10], dtype=jnp.float64)
    residuals = []
    for dt in deltas:
        times = jnp.array([t0, t0 + dt], dtype=jnp.float64)
        Sigmas = snapshot_covariances(SIGMA0, times, NU, SIGMA2, Omega)
        mus = jnp.stack([ou_mean(MU0, t, H) for t in times])
        psi = jnp.stack(
            [e_psi_sqnorm(Sigmas[k], mu=mus[k]) for k in range(2)]
        )
        gen = jnp.stack(
            [e_generator_sqnorm(H, Sigmas[k], SIGMA2, mu=mus[k]) for k in range(2)]
        )
        r = trapezoidal_residual(psi, gen, dts=jnp.array([dt]))
        residuals.append(jnp.abs(r[0]))
    r0, r1, r2 = residuals
    # O(Δ) would shrink by about 1/2 when Δ halves. O(Δ^3) shrinks by about 1/8.
    assert r1 < 0.45 * r0
    assert r2 < 0.45 * r1
    ratio = float(r1 / r0)
    assert ratio < 0.35
