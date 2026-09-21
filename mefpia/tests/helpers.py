from __future__ import annotations

import jax
import jax.numpy as jnp

from mefpia.ops import hermitize, random_density


def kron_all(mats):
    out = mats[0]
    for mat in mats[1:]:
        out = jnp.kron(out, mat)
    return out


def random_hermitian(key, dim, dtype=jnp.complex128):
    key_r, key_i = jax.random.split(key)
    real_dtype = jnp.finfo(dtype).dtype
    a = jax.random.normal(key_r, (dim, dim), dtype=real_dtype).astype(dtype)
    a = a + 1j * jax.random.normal(key_i, (dim, dim), dtype=real_dtype).astype(dtype)
    return hermitize(a)


def random_product_state(key, dims, dtype=jnp.complex128):
    keys = jax.random.split(key, len(dims))
    return [random_density(k, d, dtype=dtype) for k, d in zip(keys, dims)]
