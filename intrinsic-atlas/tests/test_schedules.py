"""Lock Appendix A coefficients and eq. (4) contraction factors."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from intrinsic_atlas.gaussian import FIG4_LAMBDAS, FIG4_RHO_PRODUCTS, FIG4_T
from intrinsic_atlas.schedules import make_vp_schedule, rho_product, rho_t


def test_c1_c2_identity_linear_schedule():
    """c_{1,t} + c_{2,t} √ᾱ_t = √ᾱ_{t-1} on a linear β schedule."""
    sched = make_vp_schedule(T=FIG4_T)
    lhs = sched.c1 + sched.c2 * jnp.sqrt(sched.alphas_cumprod)
    rhs = jnp.sqrt(sched.alphas_cumprod_prev)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-10, atol=1e-12)


def test_rho_in_unit_interval():
    """0 < ρ_t(λ) < 1 at every step for every λ > 0 (1-ᾱ_t ≥ β_t)."""
    sched = make_vp_schedule(T=FIG4_T)
    for lam in FIG4_LAMBDAS:
        rho = np.asarray(rho_t(sched, lam))
        assert np.all(rho > 0.0)
        assert np.all(rho < 1.0)
    # Also the bound 1-ᾱ_t ≥ β_t used in the proof.
    np.testing.assert_array_less(-1e-15, np.asarray(1.0 - sched.alphas_cumprod - sched.betas))


def test_rho_product_figure4():
    """∏_t ρ_t for T=1000 linear schedule, λ ∈ {0.084, 0.716} near 3.4e-6 and 2.9e-5."""
    sched = make_vp_schedule(T=FIG4_T)
    for lam, expected in zip(FIG4_LAMBDAS, FIG4_RHO_PRODUCTS):
        prod = float(rho_product(sched, lam))
        np.testing.assert_allclose(prod, expected, rtol=0.5)


def test_alphabar_T_scale():
    """A seed starts about 1/√ᾱ_T ≈ 157 from the center in clean-image units."""
    sched = make_vp_schedule(T=FIG4_T)
    scale = float(1.0 / jnp.sqrt(sched.alphas_cumprod[-1]))
    np.testing.assert_allclose(scale, 157.0, rtol=0.1)
