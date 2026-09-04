"""Reconstruction metric used in the paper's tables."""

import jax.numpy as jnp


def psnr(pred, target, data_range=None):
    """PSNR in dB: 10 * log10(range^2 / MSE).

    The paper normalizes the compared region by its own intensity range before
    computing MSE (Appendix E-A3), which is what `data_range=None` reproduces:
    the range is taken from `target`. Pass an explicit `data_range` for the
    conventional fixed-scale PSNR.
    """
    pred = jnp.asarray(pred, dtype=jnp.float64).reshape(-1)
    target = jnp.asarray(target, dtype=jnp.float64).reshape(-1)
    if data_range is None:
        data_range = jnp.max(target) - jnp.min(target)
    mse = jnp.mean((pred - target) ** 2)
    return 10.0 * jnp.log10(data_range**2 / mse)
