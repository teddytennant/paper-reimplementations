"""Affine denoiser (eq. 3) matches Tweedie on the exact Gaussian score."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from intrinsic_atlas.gaussian import (
    FIG4_LAMBDAS,
    affine_denoiser,
    clean_deviation,
    gaussian_score,
    tweedie_x0,
)
from intrinsic_atlas.sampler import posterior_mean_step
from intrinsic_atlas.schedules import make_vp_schedule, rho_t


def test_affine_denoiser_matches_tweedie():
    mu = jnp.array([0.3, -0.2])
    cov = jnp.diag(jnp.array(FIG4_LAMBDAS))
    sched = make_vp_schedule(T=1000)
    key = jax.random.PRNGKey(0)
    # Several noise levels, including near t=T and mid-schedule.
    ts = [0, 50, 200, 500, 999]
    x = jax.random.normal(key, (8, 2))
    for t in ts:
        ab = sched.alphas_cumprod[t]
        score = gaussian_score(x, ab, mu, cov)
        tweedie = tweedie_x0(x, ab, score)
        affine = affine_denoiser(x, ab, mu, cov)
        np.testing.assert_allclose(
            np.asarray(tweedie), np.asarray(affine), rtol=1e-8, atol=1e-8
        )


def test_tweedie_recovers_e_x0_given_xt_for_gaussian():
    """For x_t = √ᾱ x0 + √(1-ᾱ) ε with x0~N(μ,C), Tweedie is the MMSE denoiser."""
    mu = jnp.array([0.0, 0.0])
    cov = jnp.diag(jnp.array(FIG4_LAMBDAS))
    sched = make_vp_schedule(T=1000)
    t = 200
    ab = sched.alphas_cumprod[t]
    # Closed form eq. (3) is E[x0|xt]; Tweedie must match it at a random xt.
    xt = jnp.array([0.4, -1.1])
    score = gaussian_score(xt, ab, mu, cov)
    np.testing.assert_allclose(
        np.asarray(tweedie_x0(xt, ab, score)),
        np.asarray(affine_denoiser(xt, ab, mu, cov)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_one_step_contracts_by_rho():
    """Along an eigen-direction, e_{t-1} = ρ_t(λ) e_t after a posterior-mean step."""
    mu = jnp.array([0.1, -0.4])
    lam = jnp.array(FIG4_LAMBDAS)
    cov = jnp.diag(lam)
    sched = make_vp_schedule(T=1000)
    t = 400
    ab = sched.alphas_cumprod[t]
    e_t = jnp.array([1.7, -0.9])
    x_t = jnp.sqrt(ab) * mu + jnp.sqrt(ab) * e_t
    x0_hat = affine_denoiser(x_t, ab, mu, cov)
    x_prev = posterior_mean_step(x_t, x0_hat, sched, t)
    e_prev = clean_deviation(x_prev, sched.alphas_cumprod_prev[t], mu)
    expected = jnp.stack([rho_t(sched, lam[i])[t] * e_t[i] for i in range(2)])
    np.testing.assert_allclose(np.asarray(e_prev), np.asarray(expected), rtol=1e-10, atol=1e-12)
