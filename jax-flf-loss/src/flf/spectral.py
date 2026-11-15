"""Radial frequency bookkeeping for the 2D DFT.

Everything here works on NHWC arrays and uses ``jnp.fft`` so it stays jittable
and differentiable. The band machinery is what the paper uses for its
frequency-domain error analysis (Figure 6 right, Appendix B.2) and for the
discrete-band ablation of Table 9.
"""

from __future__ import annotations

import jax.numpy as jnp

SPATIAL_AXES = (-3, -2)


def fft2(x):
    """2D DFT over the spatial axes of an NHWC array, unnormalized.

    Unnormalized means Parseval reads ``sum |F|^2 = H * W * sum |x|^2``, which is
    the constant the focal-exponent-zero reduction in ``losses.py`` relies on.
    """
    return jnp.fft.fft2(x, axes=SPATIAL_AXES)


def radial_frequency(height: int, width: int):
    """Radial spatial frequency in cycles/pixel for every DFT bin, unshifted.

    Runs from 0 at DC to sqrt(0.5) ~ 0.707 at the corner of the spectrum; the
    Nyquist limit along an axis is 0.5.
    """
    fy = jnp.fft.fftfreq(height)[:, None]
    fx = jnp.fft.fftfreq(width)[None, :]
    return jnp.sqrt(fy**2 + fx**2)


def band_index(height: int, width: int, n_bands: int):
    """Assign each DFT bin to one of ``n_bands`` equal-width radial rings.

    Bins are cut on [0, 0.5], the axis Nyquist. The corner bins past 0.5 fall in
    the last ring.
    """
    k = radial_frequency(height, width)
    idx = jnp.floor(k / 0.5 * n_bands).astype(jnp.int32)
    return jnp.clip(idx, 0, n_bands - 1)


def band_means(values, n_bands: int):
    """Mean of an NHWC map inside each radial ring, returned as (n_bands,).

    Batch and channel axes are averaged first. Averaging per mode rather than
    summing removes the trivial fact that outer rings hold more DFT bins, so a
    flat profile means every mode carries the same weight.
    """
    height, width = values.shape[-3], values.shape[-2]
    plane = jnp.mean(values, axis=tuple(range(values.ndim - 3)) + (-1,))
    idx = band_index(height, width, n_bands).reshape(-1)
    counts = jnp.bincount(idx, length=n_bands)
    sums = jnp.bincount(idx, weights=plane.reshape(-1), length=n_bands)
    return sums / jnp.maximum(counts, 1)


def annulus_mask(height: int, width: int, low: float, high: float):
    """Boolean mask for DFT bins with low <= k < high, in cycles/pixel."""
    k = radial_frequency(height, width)
    return (k >= low) & (k < high)


def band_relative_error(pred, target, low: float, high: float):
    """Error power in an annulus, divided by the target's power there.

    1.0 is what a model gets for predicting the mean image, so this is the
    normalized form of the per-band magnitude error in Figure 6 (right) and of
    the relative spectral deviation in Figure 2(a).
    """
    mask = annulus_mask(pred.shape[-3], pred.shape[-2], low, high)[..., None]
    err = jnp.sum(jnp.abs(fft2(pred - target)) ** 2 * mask)
    ref = jnp.sum(jnp.abs(fft2(target)) ** 2 * mask)
    return err / ref
