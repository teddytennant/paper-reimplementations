"""End-to-end residual-MLP attribution conservation."""

import jax.numpy as jnp
from jax import random

from reslrp.explain import attribute
from reslrp.linear import lrp_eps
from reslrp.models import init_residual_mlp, residual_mlp


def test_residual_mlp_conservation():
    key = random.PRNGKey(0)
    params = init_residual_mlp(key, d_in=16, d_hidden=32, n_classes=4, n_blocks=2)
    x = random.normal(random.PRNGKey(1), (8, 16))
    class_idx = 2
    logits = residual_mlp(params, x)
    r = attribute(params, x, class_idx, gamma=1.0)
    err = jnp.abs(jnp.sum(r, axis=-1) - logits[:, class_idx])
    assert float(jnp.max(err)) < 1e-4


def test_residual_mlp_cache_and_energy():
    key = random.PRNGKey(2)
    params = init_residual_mlp(key, d_in=16, n_blocks=2)
    x = random.normal(random.PRNGKey(3), (4, 16))
    logits, caches = residual_mlp(params, x, cache=True)
    assert logits.shape == (4, 4)
    assert len(caches) == 2
    for c in caches:
        assert c["z_in"].shape == (4, 16)
        assert c["z_up"].shape == (4, 16)
        assert jnp.allclose(c["z_out"], c["z_in"] + c["z_up"])
    r = attribute(params, x, class_idx=0, gamma=0.5)
    energy = float(jnp.sum(r**2))
    assert energy >= 0.0


def test_linear_lrp_eps_conservation():
    key = random.PRNGKey(4)
    k1, k2, k3 = random.split(key, 3)
    x = random.normal(k1, (6, 10))
    w = random.normal(k2, (10, 5))
    r_out = random.normal(k3, (6, 5))
    r_in = lrp_eps(x, w, None, r_out, eps=1e-9)
    assert r_in.shape == (6, 10)
    assert jnp.allclose(jnp.sum(r_in, axis=-1), jnp.sum(r_out, axis=-1), atol=1e-8)


def test_attribute_batched_shape():
    params = init_residual_mlp(random.PRNGKey(5), d_in=16, n_blocks=2)
    x = random.normal(random.PRNGKey(6), (3, 16))
    r = attribute(params, x, class_idx=1, gamma=1.0)
    assert r.shape == (3, 16)
