"""Synthetic images with a controlled 1/f^2 power spectrum.

The paper's premise, stated in the abstract and the first paragraph of Section 1:
natural image power spectra fall off as 1/f^2, so most of the energy sits in the
low frequencies while the perceptually important edges and textures live in
sparse high-frequency bands. Pixel objectives weight all spatial errors equally,
so the low frequencies dominate the gradient.

These samples reproduce that structure and nothing else. A 1/f^2 background
gives the energy imbalance, and a handful of fixed high radial frequencies with
per-sample random phase gives the sparse high-frequency structure that a model
has to actually learn rather than average away.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .spectral import radial_frequency


def spectral_envelope(size: int, exponent: float = 1.0):
    """Amplitude envelope 1/k^exponent, so power falls off as 1/k^(2*exponent).

    The default gives the 1/f^2 power law. DC is set to zero so samples have
    zero mean.
    """
    k = radial_frequency(size, size)
    k = jnp.where(k > 0, k, 1.0)
    env = 1.0 / k**exponent
    return env.at[0, 0].set(0.0)


def sample_images(
    key,
    batch: int,
    size: int = 16,
    channels: int = 1,
    exponent: float = 1.0,
    hf_amplitude: float = 0.35,
    hf_modes=((5, 2), (3, 6), (7, 5)),
    dtype=jnp.float32,
):
    """Draw ``batch`` images of shape (batch, size, size, channels).

    The background is white noise shaped by ``spectral_envelope`` and rescaled
    to unit standard deviation per image. On top of it sit ``hf_modes`` cosines
    at fixed integer frequencies with independent per-sample phase and
    amplitude ``hf_amplitude``. With size 16 and the default modes, every mode
    sits above 0.3 cycles/pixel, inside the top radial band.
    """
    k_noise, k_phase = jax.random.split(key)
    noise = jax.random.normal(k_noise, (batch, size, size, channels), dtype=dtype)
    spec = jnp.fft.fft2(noise, axes=(-3, -2)) * spectral_envelope(size, exponent)[
        ..., None
    ]
    background = jnp.real(jnp.fft.ifft2(spec, axes=(-3, -2)))
    scale = jnp.std(background, axis=(-3, -2, -1), keepdims=True)
    background = background / jnp.maximum(scale, 1e-8)

    modes = jnp.asarray(hf_modes, dtype=dtype)
    phase = jax.random.uniform(
        k_phase, (batch, modes.shape[0], channels), dtype=dtype
    ) * (2.0 * jnp.pi)
    grid = jnp.arange(size, dtype=dtype)
    arg = (
        2.0
        * jnp.pi
        * (
            modes[:, 0, None, None] * grid[None, :, None] / size
            + modes[:, 1, None, None] * grid[None, None, :] / size
        )
    )
    detail = jnp.cos(arg[None, :, :, :, None] + phase[:, :, None, None, :])
    detail = hf_amplitude * jnp.sum(detail, axis=1)
    return (background + detail).astype(dtype)
