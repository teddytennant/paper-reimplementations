"""Block-averaged DPS: two identical blocks recover the single-block nu."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from gp_dps.block import average_block_sensitivities
from gp_dps.select import dps_from_params


def test_two_identical_blocks_match_single_block_nu():
    n, d = 20, 3
    key = jax.random.PRNGKey(5)
    kx, ky, k1, k2 = jax.random.split(key, 4)
    X = jax.random.normal(kx, (n, d), dtype=jnp.float64)
    y = 1.2 * X[:, 0] + 0.4 * X[:, 1] + 0.1 * jax.random.normal(ky, (n,), dtype=jnp.float64)
    ell = jnp.ones(d, dtype=jnp.float64)
    eta = jnp.array(float(jnp.var(y)), dtype=jnp.float64)
    sigma2 = jnp.array(0.06, dtype=jnp.float64)
    single = dps_from_params(X, y, ell, eta, sigma2, k1, B=16, q=0.2)
    blocked = average_block_sensitivities(
        [(X, y), (X, y)],
        ell,
        eta,
        sigma2,
        k2,
        B=16,
        q=0.2,
    )
    assert jnp.allclose(blocked["nu"], single["nu"], rtol=1e-10, atol=1e-10)
    assert jnp.allclose(blocked["nu"], jnp.mean(blocked["nu_blocks"], axis=0))
    assert blocked["nu_blocks"].shape[0] == 2
