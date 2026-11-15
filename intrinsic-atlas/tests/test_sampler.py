"""Figure 4 / Table 2: exact-score posterior mean collapses; DDIM and DDPM do not."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from intrinsic_atlas.gaussian import (
    FIG4_LAMBDAS,
    FIG4_T,
    make_gaussian_x0_fn,
)
from intrinsic_atlas.sampler import (
    sample_ddim,
    sample_posterior_mean,
    sample_stochastic,
    spread_offset,
)
from intrinsic_atlas.schedules import make_vp_schedule, rho_product


def _fig4_setup():
    mu = jnp.zeros((2,))
    cov = jnp.diag(jnp.array(FIG4_LAMBDAS))
    sched = make_vp_schedule(T=FIG4_T)
    predict_x0 = make_gaussian_x0_fn(mu, cov, sched)
    return mu, cov, sched, predict_x0


def test_posterior_mean_collapses_near_mu():
    """Many seeds end near μ; leftover offset matches ∏ ρ times the 1/√ᾱ_T seed scale."""
    mu, cov, sched, predict_x0 = _fig4_setup()
    n = 32
    key = jax.random.PRNGKey(1)
    x_T = jax.random.normal(key, (n, 2))

    def run(x):
        return sample_posterior_mean(x, predict_x0, sched)

    endpoints = jax.jit(jax.vmap(run))(x_T)
    spread, offset = spread_offset(endpoints, mu)
    spread = float(spread)
    offset = float(offset)
    mean_dist = float(jnp.mean(jnp.linalg.norm(endpoints - mu, axis=-1)))

    assert spread < 0.05
    assert offset < 0.05
    assert mean_dist < 0.05
    # Predicted leftover: seed scale 1/√ᾱ_T contracted by ∏ρ per axis.
    scale = float(1.0 / jnp.sqrt(sched.alphas_cumprod[-1]))
    pred = np.array([float(rho_product(sched, lam)) * scale for lam in FIG4_LAMBDAS])
    # Mean Euclidean leftover for N(0, diag(pred^2)) is order ||pred||.
    assert mean_dist < 5.0 * float(np.linalg.norm(pred))


def test_stochastic_and_ddim_do_not_collapse():
    """Same exact score: DDIM (η=0) and stochastic DDPM keep seed diversity."""
    mu, cov, sched, predict_x0 = _fig4_setup()
    n = 32
    key = jax.random.PRNGKey(2)
    key_x, key_stoch = jax.random.split(key)
    x_T = jax.random.normal(key_x, (n, 2))

    pm = jax.jit(jax.vmap(lambda x: sample_posterior_mean(x, predict_x0, sched)))(x_T)
    ddim = jax.jit(jax.vmap(lambda x: sample_ddim(x, predict_x0, sched)))(x_T)

    def stoch_one(x, k):
        return sample_stochastic(x, predict_x0, sched, k)

    keys = jax.random.split(key_stoch, n)
    stoch = jax.jit(jax.vmap(stoch_one))(x_T, keys)

    pm_spread, _ = spread_offset(pm, mu)
    ddim_spread, _ = spread_offset(ddim, mu)
    stoch_spread, _ = spread_offset(stoch, mu)

    pm_spread = float(pm_spread)
    ddim_spread = float(ddim_spread)
    stoch_spread = float(stoch_spread)

    # Table 2(a): posterior mean ~0.004 vs stochastic ~0.74 / DDIM ~0.73.
    assert pm_spread < 0.05
    assert stoch_spread > 0.3
    assert ddim_spread > 0.3
    assert pm_spread < 0.1 * stoch_spread
    assert pm_spread < 0.1 * ddim_spread


def test_ddim_is_not_posterior_mean():
    """A single step disagrees: DDIM re-inserts ε̂, posterior mean does not."""
    from intrinsic_atlas.gaussian import affine_denoiser
    from intrinsic_atlas.sampler import ddim_step, posterior_mean_step

    mu, cov, sched, _ = _fig4_setup()
    t = 500
    x_t = jnp.array([0.8, -0.3])
    x0 = affine_denoiser(x_t, sched.alphas_cumprod[t], mu, cov)
    pm = posterior_mean_step(x_t, x0, sched, t)
    ddim = ddim_step(x_t, x0, sched, t)
    assert float(jnp.linalg.norm(pm - ddim)) > 1e-3
