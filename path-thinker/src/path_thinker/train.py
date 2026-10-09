"""x0 loss only. Painter and Thinker updates do not touch each other's weights."""

import jax
import jax.numpy as jnp

from path_thinker.painter import painter_forward
from path_thinker.thinker import decode, recurse, recurse_from


def alpha_bars(n=100, beta_start=1e-4, beta_end=0.02):
    betas = jnp.linspace(beta_start, beta_end, n)
    return jnp.cumprod(1.0 - betas)


def q_sample(x0, eps, ab_t):
    return jnp.sqrt(ab_t) * x0 + jnp.sqrt(1.0 - ab_t) * eps


def x0_loss(pred, x0):
    return jnp.mean((x0 - pred) ** 2)


def diffusion_loss(painter, x_t, x0, t, control=None, adapters=None):
    pred = painter_forward(painter, x_t, t, control=control, adapters=adapters)
    return x0_loss(pred, x0)


def cfg(cond, uncond, scale):
    return uncond + scale * (cond - uncond)


def _sgd(params, grads, lr):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def train_painter_step(painter, thinker, batch, lr):
    """Update the Painter on a no-control x0 loss. Thinker weights stay put."""

    def loss_fn(pp):
        return diffusion_loss(pp, batch["x_t"], batch["x0"], batch["t"], control=None)

    grads = jax.grad(loss_fn)(painter)
    return _sgd(painter, grads, lr), thinker


def _thinker_loss(painter, thinker, batch, n_sup, H, L, n_spatial, height, width):
    ys, _zs = recurse(thinker, batch["c"], n_sup, H, L)
    total = 0.0
    for y in ys:
        ctrl = decode(thinker, y, n_spatial, height, width)
        total = total + diffusion_loss(
            painter,
            batch["x_t"],
            batch["x0"],
            batch["t"],
            control=ctrl,
            adapters=thinker,
        )
    return total


def thinker_loss(painter, thinker, batch, n_sup, H, L, n_spatial, height, width):
    return _thinker_loss(painter, thinker, batch, n_sup, H, L, n_spatial, height, width)


def train_thinker_step(painter, thinker, batch, lr, n_sup, H, L, n_spatial, height, width):
    """Sum the x0 loss at each supervision step. Painter weights are frozen."""
    painter_frozen = jax.tree.map(jax.lax.stop_gradient, painter)

    def loss_fn(tp):
        return _thinker_loss(
            painter_frozen, tp, batch, n_sup, H, L, n_spatial, height, width
        )

    grads = jax.grad(loss_fn)(thinker)
    return painter, _sgd(thinker, grads, lr)


def sample(
    painter,
    thinker,
    x_T,
    c,
    n_steps,
    n_sup,
    H,
    L,
    n_spatial,
    height,
    width,
    mode="default",
    on_step=None,
):
    """Denoise for n_steps.

    default: reset y and z to the learned init at every step.
    efficient: carry (y, z), run the Thinker on the first half of steps, and
    reuse that last control array for the rest.
    """
    x = x_T
    b = x.shape[0]
    y = jnp.broadcast_to(thinker["y0"], (b,) + thinker["y0"].shape)
    z = jnp.broadcast_to(thinker["z0"], (b,) + thinker["z0"].shape)
    control = None
    half = n_steps // 2
    for i in range(n_steps):
        t = jnp.full((b,), n_steps - 1 - i)
        if mode == "default":
            y = jnp.broadcast_to(thinker["y0"], (b,) + thinker["y0"].shape)
            z = jnp.broadcast_to(thinker["z0"], (b,) + thinker["z0"].shape)
        z_in = z
        if mode == "default" or i < half:
            ys, zs = recurse_from(thinker, c, y, z, n_sup, H, L)
            y, z = ys[-1], zs[-1]
            control = decode(thinker, y, n_spatial, height, width)
        if on_step is not None:
            on_step(i, z_in, control)
        pred = painter_forward(painter, x, t, control=control, adapters=thinker)
        x = pred if i == n_steps - 1 else 0.5 * x + 0.5 * pred
    return x
