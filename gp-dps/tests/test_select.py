"""BY power check and BH vs BY liberalism."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from gp_dps.select import bh_reject, by_reject, dps_from_params


def test_by_selects_linear_signal_among_noise_inputs():
    n, d = 50, 3
    key = jax.random.PRNGKey(11)
    kx, ky, kb = jax.random.split(key, 3)
    X = jax.random.normal(kx, (n, d), dtype=jnp.float64)
    y = 2.4 * X[:, 0] + 0.08 * jax.random.normal(ky, (n,), dtype=jnp.float64)
    ell = jnp.ones(d, dtype=jnp.float64)
    eta = jnp.array(float(jnp.var(y)), dtype=jnp.float64)
    sigma2 = jnp.array(0.04, dtype=jnp.float64)
    out = dps_from_params(X, y, ell, eta, sigma2, kb, B=48, q=0.2, procedure="by")
    selected = set(int(i) for i in np.asarray(out["selected"]))
    assert 0 in selected


def test_bh_is_more_liberal_than_by():
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.2, 0.5, 0.8])
    q = 0.2
    bh = bh_reject(p, q)
    by = by_reject(p, q)
    assert int(bh.sum()) >= int(by.sum())
    assert np.all(by <= bh)

    n, d = 40, 4
    key = jax.random.PRNGKey(3)
    kx, ky, kb = jax.random.split(key, 3)
    X = jax.random.normal(kx, (n, d), dtype=jnp.float64)
    y = 2.0 * X[:, 0] + 0.1 * jax.random.normal(ky, (n,), dtype=jnp.float64)
    ell = jnp.ones(d, dtype=jnp.float64)
    eta = jnp.array(float(jnp.var(y)), dtype=jnp.float64)
    sigma2 = jnp.array(0.05, dtype=jnp.float64)
    out = dps_from_params(X, y, ell, eta, sigma2, kb, B=32, q=0.2, procedure="by")
    pvals = np.asarray(out["p"])
    assert int(bh_reject(pvals, 0.2).sum()) >= int(by_reject(pvals, 0.2).sum())
