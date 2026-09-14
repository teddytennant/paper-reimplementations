"""Position-wise factorization of the projected discrete interpolant."""

import jax
import jax.numpy as jnp

from flow_duality.coefficients import k_gaussian
from flow_duality.interpolant import argmax_tokens, convex_interpolant, one_hot_tokens
from flow_duality.sources import sample_gaussian


def test_factorization_l3():
    v = 2
    l = 3
    n = 20000
    t = 0.5
    key = jax.random.PRNGKey(4)
    z0 = sample_gaussian(key, (n, l, v))
    x0 = argmax_tokens(z0)
    x1 = (x0 + 1) % v
    z1 = one_hot_tokens(x1, v, dtype=z0.dtype)
    zt = convex_interpolant(z0, z1, t)
    xt = argmax_tokens(zt)
    flipped = (xt == x1).astype(jnp.int32)
    codes = flipped[:, 0] * 4 + flipped[:, 1] * 2 + flipped[:, 2]
    emp = jnp.bincount(codes, length=8).astype(jnp.float32) / n

    kt = float(k_gaussian(t, v))
    prod = []
    for s in range(8):
        bits = ((s >> 2) & 1, (s >> 1) & 1, s & 1)
        p = 1.0
        for b in bits:
            p *= kt if b else (1.0 - kt)
        prod.append(p)
    prod = jnp.asarray(prod)
    l1 = float(jnp.sum(jnp.abs(emp - prod)))
    assert l1 < 0.05, (l1, emp, prod, kt)
