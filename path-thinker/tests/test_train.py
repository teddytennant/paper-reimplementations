"""x0 loss ignores eps given x_t. Updates do not cross the freeze."""

import inspect

import jax
import jax.numpy as jnp

from path_thinker.painter import init_painter, painter_forward
from path_thinker.thinker import init_thinker
from path_thinker.train import (
    alpha_bars,
    diffusion_loss,
    q_sample,
    train_painter_step,
    train_thinker_step,
    x0_loss,
)


def _max_delta(a, b):
    leaves = jax.tree.map(lambda u, v: jnp.max(jnp.abs(u - v)), a, b)
    return max(float(x) for x in jax.tree.leaves(leaves))


def _batch():
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    painter = init_painter(k1, c_in=1, scale=0.2)
    thinker = init_thinker(k2, d=8, y_len=5, z_len=2, c_skip=8, scale=0.2)
    x0 = jax.random.normal(k3, (2, 2, 2, 1))
    eps = jax.random.normal(k4, x0.shape)
    t = jnp.array([10, 20])
    ab = alpha_bars(100)
    ab_t = ab[t][:, None, None, None]
    x_t = q_sample(x0, eps, ab_t)
    c = jax.random.normal(jax.random.PRNGKey(7), (2, 3, 8))
    batch = {"x0": x0, "x_t": x_t, "t": t, "c": c, "eps": eps, "ab_t": ab_t}
    return painter, thinker, batch


def test_loss_is_x0_and_ignores_eps_given_xt():
    painter, _thinker, batch = _batch()
    pred = painter_forward(painter, batch["x_t"], batch["t"])
    assert jnp.allclose(x0_loss(pred, batch["x0"]), jnp.mean((batch["x0"] - pred) ** 2))
    assert "eps" not in inspect.signature(diffusion_loss).parameters

    loss_xt = diffusion_loss(painter, batch["x_t"], batch["x0"], batch["t"])
    assert jnp.allclose(loss_xt, jnp.mean((batch["x0"] - pred) ** 2))

    eps2 = jnp.ones_like(batch["eps"]) * 3.0
    xt2 = q_sample(batch["x0"], eps2, batch["ab_t"])
    assert not jnp.allclose(xt2, batch["x_t"])
    assert jnp.allclose(
        diffusion_loss(painter, batch["x_t"], batch["x0"], batch["t"]),
        loss_xt,
    )
    assert not jnp.allclose(
        diffusion_loss(painter, xt2, batch["x0"], batch["t"]),
        loss_xt,
    )


def test_updates_do_not_cross():
    painter, thinker, batch = _batch()
    new_p, new_t = train_painter_step(painter, thinker, batch, lr=0.05)
    assert _max_delta(new_t, thinker) == 0.0
    assert _max_delta(new_p, painter) > 0.0

    new_p2, new_t2 = train_thinker_step(
        painter, thinker, batch,
        lr=0.05, n_sup=1, H=1, L=1,
        n_spatial=4, height=2, width=2,
    )
    assert _max_delta(new_p2, painter) == 0.0
    assert _max_delta(new_t2, thinker) > 0.0
