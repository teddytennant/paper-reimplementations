"""Kernel derivatives vs finite differences (Appendix A) and prior deriv variance."""
from __future__ import annotations

import jax.numpy as jnp

from gp_dps.kernel import prior_deriv_variance, se_ard, se_ard_deriv_first


def _params():
    ell = jnp.array([0.7, 1.2, 0.4], dtype=jnp.float64)
    eta = jnp.array(1.7, dtype=jnp.float64)
    return ell, eta


def test_kernel_deriv_matches_finite_difference():
    ell, eta = _params()
    x = jnp.array([[0.2, -0.4, 0.9], [1.1, 0.3, -0.7]], dtype=jnp.float64)
    x2 = jnp.array([[0.0, 0.1, 0.2], [-0.5, 0.8, 1.0], [0.4, -1.0, 0.3]], dtype=jnp.float64)
    eps = 1e-6
    for j in range(3):
        analytic = se_ard_deriv_first(x, x2, ell, eta, j)
        xp = x.at[:, j].add(eps)
        xm = x.at[:, j].add(-eps)
        fd = (se_ard(xp, x2, ell, eta) - se_ard(xm, x2, ell, eta)) / (2.0 * eps)
        assert jnp.max(jnp.abs(analytic - fd)) < 1e-5


def test_prior_diagonal_deriv_variance_equals_eta_over_ellsq():
    ell, eta = _params()
    got = prior_deriv_variance(ell, eta)
    expect = eta / (ell ** 2)
    assert jnp.allclose(got, expect)

    # Mixed second derivative of k at x=x' equals eta/ell_j^2.
    x0 = jnp.array([[0.3, -0.2, 0.5]], dtype=jnp.float64)
    eps = 1e-5
    for j in range(3):
        def k_off(dx, dxp):
            a = x0.at[0, j].add(dx)
            b = x0.at[0, j].add(dxp)
            return se_ard(a, b, ell, eta)[0, 0]

        mixed = (
            k_off(eps, eps) - k_off(eps, -eps) - k_off(-eps, eps) + k_off(-eps, -eps)
        ) / (4.0 * eps * eps)
        assert abs(float(mixed) - float(expect[j])) < 1e-4
