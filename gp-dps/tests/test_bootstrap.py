"""LOO residual identity and multiplier-bootstrap p-values on a null coordinate."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from gp_dps.bootstrap import loo_residuals
from gp_dps.gp import alpha_from_chol, cho_kn, fit_se_ard, posterior_mean
from gp_dps.select import dps_from_params


def test_loo_residual_matches_leave_one_out():
    n, d = 6, 2
    key = jax.random.PRNGKey(1)
    X = jax.random.normal(key, (n, d), dtype=jnp.float64)
    ell = jnp.array([0.9, 1.1], dtype=jnp.float64)
    eta = jnp.array(1.0, dtype=jnp.float64)
    sigma2 = jnp.array(0.08, dtype=jnp.float64)
    y = 0.6 * X[:, 0] + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (n,), dtype=jnp.float64)
    L = cho_kn(X, ell, eta, sigma2)
    r = loo_residuals(L, y)
    alpha = alpha_from_chol(L, y)
    from gp_dps.gp import kn_inv_diag

    inv_diag = kn_inv_diag(L)
    assert jnp.allclose(r, alpha / inv_diag)

    # Explicit leave-one-out residual with frozen hyperparameters.
    loo = []
    for i in range(n):
        mask = jnp.ones(n, dtype=bool).at[i].set(False)
        Xi, yi = X[mask], y[mask]
        Li = cho_kn(Xi, ell, eta, sigma2)
        ai = alpha_from_chol(Li, yi)
        pred = posterior_mean(Xi, ai, X[i : i + 1], ell, eta)[0]
        loo.append(y[i] - pred)
    loo = jnp.array(loo)
    assert jnp.allclose(r, loo, rtol=1e-5, atol=1e-5)


def test_null_coordinate_pvalues_not_all_tiny():
    n, d = 36, 3
    key = jax.random.PRNGKey(7)
    kx, ky, kb = jax.random.split(key, 3)
    X = jax.random.normal(kx, (n, d), dtype=jnp.float64)
    # Only coordinate 0 is active; 1 and 2 are pure-noise inputs.
    y = 1.8 * X[:, 0] + 0.15 * jax.random.normal(ky, (n,), dtype=jnp.float64)
    ell, eta, sigma2 = fit_se_ard(X, y, steps=80, lr=0.08)
    out = dps_from_params(X, y, ell, eta, sigma2, kb, B=64, q=0.2, procedure="by")
    p_null = jnp.asarray(out["p"])[1:]
    assert float(jnp.min(p_null)) > 1e-6
    assert float(jnp.mean(p_null)) >= 0.05
