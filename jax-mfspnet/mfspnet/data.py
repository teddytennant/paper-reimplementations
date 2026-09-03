"""Tiny synthetic image task and the downscaled proxy set (paper Sec. 4.4).

The paper searches on CIFAR-10 downscaled by n = 4, i.e. 8x8, batch size 256,
Adam at 1e-3, 15 epochs per candidate (Table 3). Nothing here downloads
CIFAR-10. :func:`synthetic_task` makes a small classification problem out of
oriented gratings plus a class-dependent color bias, which is learnable by a
few conv layers and hard enough that architecture choice moves the loss.

:func:`downscale` is the proxy-dataset transform of Sec. 4.4: average pool by
n so a w x h image becomes w/n x h/n.

On the factor itself the paper contradicts itself. Table 2, Sec. 4.4 and
Appendix B all say n = 4, and Table 3 gives the search resolution as 8x8, which
is 32/4. Sec. 5.3 says 2 in passing. ``DOWNSCALE_FACTOR`` is 4, the value three
of the four places agree on.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

DOWNSCALE_FACTOR = 4


@dataclasses.dataclass(frozen=True)
class Dataset:
    x_train: jnp.ndarray
    y_train: jnp.ndarray
    x_val: jnp.ndarray
    y_val: jnp.ndarray

    @property
    def n_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_val.max())) + 1

    @property
    def in_channels(self) -> int:
        return int(self.x_train.shape[-1])


def downscale(x: jnp.ndarray, n: int = DOWNSCALE_FACTOR) -> jnp.ndarray:
    """Average-pool an NHWC batch by a factor of n on both spatial axes."""
    b, h, w, c = x.shape
    if h % n or w % n:
        raise ValueError(f"{h}x{w} is not divisible by n={n}")
    return x.reshape(b, h // n, n, w // n, n, c).mean(axis=(2, 4))


def synthetic_task(
    key,
    n_train: int = 192,
    n_val: int = 96,
    size: int = 8,
    n_classes: int = 4,
    n_channels: int = 3,
    noise: float = 0.35,
) -> Dataset:
    """Oriented gratings with a per-class color bias, at ``size`` x ``size``.

    Class c gets grating angle ``pi c / n_classes`` and boosts one color
    channel. The grating is zero mean, so the color cue shows up as per-channel
    contrast, not as a per-channel offset, and a model that only looks at
    channel means learns nothing. Phase is random per image, so the angle has
    to be read from the spatial pattern.
    """
    total = n_train + n_val
    k_lab, k_phase, k_noise = jax.random.split(key, 3)
    labels = jax.random.randint(k_lab, (total,), 0, n_classes)
    phase = jax.random.uniform(k_phase, (total, 1, 1)) * 2 * jnp.pi

    grid = jnp.linspace(-1.0, 1.0, size)
    yy, xx = jnp.meshgrid(grid, grid, indexing="ij")
    angles = jnp.pi * labels / n_classes
    proj = (
        jnp.cos(angles)[:, None, None] * xx[None]
        + jnp.sin(angles)[:, None, None] * yy[None]
    )
    grating = jnp.sin(4.0 * jnp.pi * proj + phase)

    color = jax.nn.one_hot(labels % n_channels, n_channels)[:, None, None, :]
    x = grating[..., None] * (0.5 + 0.5 * color)
    x = x + noise * jax.random.normal(k_noise, x.shape)

    return Dataset(
        x_train=x[:n_train],
        y_train=labels[:n_train],
        x_val=x[n_train:],
        y_val=labels[n_train:],
    )
