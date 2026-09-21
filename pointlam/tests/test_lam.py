import jax
import jax.numpy as jnp
import numpy as np

from pointlam.lam import apply_lam_block, apply_pointlam, init_lam_block, init_pointlam


def test_lam_block_finite_shapes_grads():
    key = jax.random.PRNGKey(0)
    n, dim = 24, 8
    k1, k2, k3 = jax.random.split(key, 3)
    points = jax.random.uniform(k1, (n, 3))
    features = jax.random.normal(k2, (n, dim))
    params = init_lam_block(k3, dim, state=8)
    out = apply_lam_block(params, points, features, cell_size=0.35)
    assert out.shape == (n, dim)
    assert bool(jnp.isfinite(out).all())

    def loss_fn(p):
        return jnp.mean(apply_lam_block(p, points, features, 0.35) ** 2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree.leaves(grads)
    assert leaves, "no parameters"
    assert all(jnp.isfinite(g).all() for g in leaves)
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)


def test_no_nans_on_128_point_cloud():
    key = jax.random.PRNGKey(7)
    n, dim, m = 128, 16, 32
    k1, k2 = jax.random.split(key)
    points = jax.random.uniform(k1, (n, 3))
    params = init_pointlam(k2, dim=dim, n_blocks=2, state=8)
    pts, feats, idx, logits = apply_pointlam(
        params, points, jax.random.fold_in(key, 9), m=m, cell_size=0.2
    )
    assert pts.shape == (m, 3)
    assert feats.shape == (m, dim)
    assert idx.shape == (m,)
    assert logits.shape == (m, 1)
    for arr in (pts, feats, logits):
        assert bool(jnp.isfinite(arr).all())
        assert not bool(jnp.any(jnp.isnan(arr)))


def test_pointlam_grad_not_all_zero():
    key = jax.random.PRNGKey(11)
    n, dim, m = 32, 8, 16
    k1, k2 = jax.random.split(key)
    points = jax.random.uniform(k1, (n, 3))
    params = init_pointlam(k2, dim=dim, n_blocks=1, state=8)

    def loss_fn(p):
        _pts, _feats, _idx, logits = apply_pointlam(
            p, points, jax.random.PRNGKey(0), m=m, cell_size=0.3
        )
        return jnp.mean(logits**2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree.leaves(grads)
    assert all(jnp.isfinite(g).all() for g in leaves)
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)
