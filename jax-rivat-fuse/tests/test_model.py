"""End-to-end fuse: pytree grads, ridge identity, batch vs one."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rivat_fuse.model import fuse, fuse_one, init_params, predict
from rivat_fuse.solve import assemble_H, assemble_b, solve_latent


D = 6


def test_fuse_ridge_identity_override():
    """With P_I=P_M=I and T=0 the closed form is the ridge average."""
    params = init_params(jax.random.PRNGKey(0), d=D, lam=0.25)
    z_I = jnp.arange(D, dtype=jnp.float32)
    z_M = 2.0 * jnp.ones((D,), dtype=jnp.float32)
    P = jnp.eye(D)
    t = jnp.zeros((D,))
    lam = 0.25
    h = solve_latent(assemble_H(P, P, lam), assemble_b(P, P, z_I, z_M, t))
    assert jnp.allclose(h, (z_I + z_M) / (2.0 + lam), atol=1e-5)


def test_fuse_is_finite_and_d_dimensional():
    params = init_params(jax.random.PRNGKey(1), d=D, n_classes=2)
    z_I = jax.random.normal(jax.random.PRNGKey(2), (D,))
    z_M = jax.random.normal(jax.random.PRNGKey(3), (D,))
    h = fuse_one(params, z_I, z_M)
    assert h.shape == (D,)
    assert jnp.all(jnp.isfinite(h))
    scores = predict(params, z_I, z_M)
    assert scores.shape == (2,)


def test_fuse_with_tokens():
    params = init_params(jax.random.PRNGKey(4), d=D)
    z_I = jax.random.normal(jax.random.PRNGKey(5), (D,))
    z_M = jax.random.normal(jax.random.PRNGKey(6), (D,))
    tokens = jax.random.normal(jax.random.PRNGKey(7), (5, D))
    h = fuse_one(params, z_I, z_M, tokens=tokens)
    h0 = fuse_one(params, z_I, z_M, tokens=None)
    assert h.shape == (D,)
    assert not jnp.allclose(h, h0, atol=1e-4)


def test_batch_fuse_matches_vmap():
    params = init_params(jax.random.PRNGKey(8), d=D)
    B = 4
    z_I = jax.random.normal(jax.random.PRNGKey(9), (B, D))
    z_M = jax.random.normal(jax.random.PRNGKey(10), (B, D))
    h_b = fuse(params, z_I, z_M)
    h_one = jax.vmap(lambda a, b: fuse_one(params, a, b))(z_I, z_M)
    assert jnp.allclose(h_b, h_one, atol=1e-5)


def test_grad_through_fuse():
    params = init_params(jax.random.PRNGKey(11), d=D, n_classes=1)
    z_I = jax.random.normal(jax.random.PRNGKey(12), (D,))
    z_M = jax.random.normal(jax.random.PRNGKey(13), (D,))

    def loss(p):
        return jnp.sum(predict(p, z_I, z_M) ** 2)

    g = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(g)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in leaves if leaf.size)
