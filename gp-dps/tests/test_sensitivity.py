"""Eqs 2--4: posterior derivative mean/var and the sensitivity estimator."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from gp_dps.gp import alpha_from_chol, cho_kn, posterior_cov, posterior_mean
from gp_dps.kernel import prior_deriv_variance, se_ard_deriv_first
from gp_dps.sensitivity import (
    A_maps,
    all_deriv_kernel_train,
    deriv_mean_at,
    deriv_mean_train,
    deriv_var_at,
    deriv_var_train,
    dps_maps,
    plugin_quadratic,
    sensitivity_nu,
)


def _toy_gp(n=12, d=3, seed=0):
    key = jax.random.PRNGKey(seed)
    kx, ky = jax.random.split(key)
    X = jax.random.uniform(kx, (n, d), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    ell = jnp.array([0.8, 1.1, 0.6], dtype=jnp.float64)
    eta = jnp.array(1.3, dtype=jnp.float64)
    sigma2 = jnp.array(0.05, dtype=jnp.float64)
    y = 1.4 * X[:, 0] - 0.7 * X[:, 1] + 0.05 * jax.random.normal(ky, (n,), dtype=jnp.float64)
    return X, y, ell, eta, sigma2


def test_posterior_mean_derivative_matches_finite_diff_eq2():
    X, y, ell, eta, sigma2 = _toy_gp()
    L = cho_kn(X, ell, eta, sigma2)
    alpha = alpha_from_chol(L, y)
    x = jnp.array([0.15, -0.3, 0.4], dtype=jnp.float64)
    eps = 1e-5
    for j in range(X.shape[1]):
        analytic = deriv_mean_at(X, alpha, x, ell, eta, j)[0]
        xp = x.at[j].add(eps)
        xm = x.at[j].add(-eps)
        fp = posterior_mean(X, alpha, xp[None, :], ell, eta)[0]
        fm = posterior_mean(X, alpha, xm[None, :], ell, eta)[0]
        fd = (fp - fm) / (2.0 * eps)
        assert abs(float(analytic - fd)) < 1e-5


def test_posterior_deriv_variance_nonneg_and_gp_identity_eq3():
    X, y, ell, eta, sigma2 = _toy_gp()
    L = cho_kn(X, ell, eta, sigma2)
    x = jnp.array([0.1, 0.2, -0.15], dtype=jnp.float64)
    eps = 1e-4
    for j in range(X.shape[1]):
        v = deriv_var_at(X, L, x, ell, eta, j)[0]
        assert float(v) >= -1e-10

        djk = se_ard_deriv_first(x[None, :], X, ell, eta, j)[0]
        from jax.scipy.linalg import cho_solve

        quad = djk @ cho_solve((L, True), djk)
        identity = prior_deriv_variance(ell, eta)[j] - quad
        assert abs(float(v - identity)) < 1e-10

        # Mixed second finite difference of the posterior covariance (Appendix A).
        def c(dx, dxp):
            a = x.at[j].add(dx)[None, :]
            b = x.at[j].add(dxp)[None, :]
            Xs = jnp.concatenate([a, b], axis=0)
            C = posterior_cov(X, L, Xs, ell, eta)
            return C[0, 1]

        mixed = (c(eps, eps) - c(eps, -eps) - c(-eps, eps) + c(-eps, -eps)) / (4.0 * eps * eps)
        assert abs(float(v) - float(mixed)) < 5e-4


def test_sensitivity_is_plugin_plus_mean_posterior_var_eq4():
    X, y, ell, eta, sigma2 = _toy_gp()
    maps = dps_maps(X, y, ell, eta, sigma2)
    nu = sensitivity_nu(X, y, ell, eta, sigma2)
    plugin = jnp.mean(maps["gbar"] ** 2, axis=-1)
    mean_v = jnp.mean(maps["v"], axis=-1)
    expect = plugin + mean_v
    assert jnp.allclose(nu, expect)
    assert jnp.allclose(nu, maps["plugin"] + mean_v)
    assert jnp.allclose(maps["plugin"], plugin_quadratic(maps["A"], y))


def test_linear_sine_sensitivity_order_of_magnitude():
    """f(x)=a1 x1 + a2 sin(omega x2): nu_1 ~ a1^2, nu_2 ~ a2^2 omega^2 E[cos^2]."""
    a1, a2, omega = 1.5, 0.8, 1.7
    g1 = 12
    g2 = 12
    t = jnp.linspace(-1.0, 1.0, g1, dtype=jnp.float64)
    s = jnp.linspace(-1.0, 1.0, g2, dtype=jnp.float64)
    xx, yy = jnp.meshgrid(t, s, indexing="ij")
    X = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    y = a1 * X[:, 0] + a2 * jnp.sin(omega * X[:, 1])
    ell = jnp.array([1.4, 0.7], dtype=jnp.float64)
    eta = jnp.array(float(jnp.var(y)), dtype=jnp.float64)
    sigma2 = jnp.array(1e-6, dtype=jnp.float64)
    nu = sensitivity_nu(X, y, ell, eta, sigma2)
    e_cos2 = float(jnp.mean(jnp.cos(omega * X[:, 1]) ** 2))
    target = jnp.array([a1 ** 2, (a2 ** 2) * (omega ** 2) * e_cos2], dtype=jnp.float64)
    # Ranking and order of magnitude, not paper tables.
    assert float(nu[0]) > 0.5 * float(target[0])
    assert float(nu[1]) > 0.3 * float(target[1])
    assert float(nu[0]) > float(nu[1]) or abs(float(nu[0] / target[0]) - 1.0) < 0.6
    rel = jnp.abs(nu - target) / jnp.maximum(target, 1e-6)
    assert jnp.all(rel < 1.5)
