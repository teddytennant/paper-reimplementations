"""Zero-init control, CFG, sampler reset and reuse."""

import jax
import jax.numpy as jnp

from path_thinker.painter import painter_forward, init_painter
from path_thinker.thinker import decode, init_thinker, recurse, recurse_from
from path_thinker.train import cfg, sample


def test_zero_init_then_residual():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    painter = init_painter(k1, c_in=1)
    thinker = init_thinker(k2, d=8, y_len=65, z_len=2, c_skip=8, scale=0.05)
    x = jax.random.normal(k3, (1, 8, 8, 1))
    t = jnp.array([3])
    c = jax.random.normal(jax.random.PRNGKey(4), (1, 4, 8))
    ys, _zs = recurse(thinker, c, n_sup=1, H=1, L=1)
    ctrl = decode(thinker, ys[-1], n_spatial=64, height=8, width=8)

    out_nc = painter_forward(painter, x, t, control=None)
    out_c = painter_forward(painter, x, t, control=ctrl, adapters=thinker)
    assert jnp.allclose(out_nc, out_c)

    adapters = dict(thinker)
    adapters["W_a1"] = jnp.ones_like(thinker["W_a1"]) * 0.25
    _pred, skip_nc, _lvl = painter_forward(painter, x, t, control=None, return_skips=True)
    _pred, skip_c, _lvl = painter_forward(
        painter, x, t, control=ctrl, adapters=adapters, return_skips=True
    )
    residual = ctrl @ adapters["W_a1"] + adapters["b_a1"]
    assert jnp.allclose(skip_c, skip_nc + residual)


def test_cfg_is_uncond_plus_scale_times_diff():
    cond = jnp.array([1.0, 2.0])
    uncond = jnp.array([0.5, 0.0])
    assert jnp.allclose(cfg(cond, uncond, 2.5), uncond + 2.5 * (cond - uncond))


def test_default_sampler_resets_z():
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    painter = init_painter(k1, c_in=1, scale=0.2)
    thinker = init_thinker(k2, d=8, y_len=5, z_len=3, c_skip=8, scale=0.3)
    x_T = jax.random.normal(k3, (1, 2, 2, 1))
    c = jax.random.normal(jax.random.PRNGKey(5), (1, 3, 8))
    y0 = jnp.broadcast_to(thinker["y0"], (1,) + thinker["y0"].shape)
    z0 = jnp.broadcast_to(thinker["z0"], (1,) + thinker["z0"].shape)
    _ys, zs = recurse_from(thinker, c, y0, z0, n_sup=1, H=1, L=1)
    assert not jnp.allclose(zs[-1], z0)

    seen = []

    def on_step(i, z_in, control):
        seen.append(z_in)

    sample(
        painter, thinker, x_T, c,
        n_steps=3, n_sup=1, H=1, L=1,
        n_spatial=4, height=2, width=2,
        mode="default", on_step=on_step,
    )
    assert jnp.allclose(seen[1], z0)
    assert not jnp.allclose(seen[1], zs[-1])


def test_efficient_sampler_reuses_control():
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    painter = init_painter(k1, c_in=1)
    thinker = init_thinker(k2, d=8, y_len=5, z_len=2, c_skip=8, scale=0.1)
    x_T = jax.random.normal(k3, (1, 2, 2, 1))
    c = jax.random.normal(jax.random.PRNGKey(6), (1, 3, 8))
    seen = []

    def on_step(i, z_in, control):
        seen.append(control)

    sample(
        painter, thinker, x_T, c,
        n_steps=4, n_sup=1, H=1, L=1,
        n_spatial=4, height=2, width=2,
        mode="efficient", on_step=on_step,
    )
    assert seen[1] is seen[2]
    assert seen[1] is seen[3]
    assert seen[0] is not seen[1]
