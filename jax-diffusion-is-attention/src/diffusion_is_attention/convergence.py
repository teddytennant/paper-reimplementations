"""Sec. II-D, what the closed form does as sigma sweeps downward.

The reverse process feeds each denoised output back in at a smaller sigma. Large
sigma flattens the differences between the ||y - s_i||, so the first outputs sit
near the dataset mean. Small sigma sharpens them, and the iterate ends up on a
sparse, near-nearest-neighbour combination. Figure 2 is the 2D three-point
version of that, reproduced by `figure2_setup` and `reverse_attention_trajectory`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .closed_form import attention_weights, optimal_denoiser

__all__ = [
    "sigma_schedule",
    "Trajectory",
    "reverse_attention_trajectory",
    "weight_entropy",
    "figure2_setup",
]

# Digitized from the axes of Figure 2 in the paper. The three signals, the query
# y, and the mean M are read off the label boxes in the PDF, so they carry the
# offset between a marker and its text label, roughly 0.05 to 0.1 in figure
# units. The published run ends near s1; this one ends near s2, which is what
# these coordinates give, since s2 is nearer the mean than s1 by about that same
# margin. The qualitative claim of Sec. II-D holds either way.
_FIGURE2_DATA = ((-0.46, 0.71), (-0.83, -0.49), (0.76, -0.50))
_FIGURE2_QUERY = (0.53, 0.24)


def sigma_schedule(sigma_start=1.0, ratio=0.96, num_steps=25):
    """The decreasing sigma sequence of Sec. II-D, sigma_t = 0.96 sigma_{t+1}.

    Figure 2 runs 25 steps from sigma = 1 down to 0.96**24 = 0.375, which is the
    "decreased gradually from 1 to 0.37" in the text.
    """
    return sigma_start * ratio ** jnp.arange(num_steps)


class Trajectory(NamedTuple):
    """`points` is (T + 1, d) including the start, `weights` is (T, N)."""

    points: jnp.ndarray
    weights: jnp.ndarray


def reverse_attention_trajectory(y, data, sigmas):
    """Iterate eq (11) with a decreasing sigma, Sec. II-D.

    z_{t+1} = z*(z_t; sigma_t). This is the paper's picture of the reverse
    process: one closed-form attention step per noise level, output fed back in.
    """
    y = jnp.asarray(y)
    data = jnp.asarray(data)
    points = [y]
    weights = []
    z = y
    for sigma in sigmas:
        weights.append(attention_weights(z, data, sigma))
        z = optimal_denoiser(z, data, sigma)
        points.append(z)
    return Trajectory(jnp.stack(points), jnp.stack(weights))


def weight_entropy(weights):
    """Shannon entropy in nats of each weight vector, over the last axis.

    Sec. II-D's "sparse representation" claim, made measurable. At fixed y this
    is a Gibbs distribution at temperature sigma^2, so entropy is monotone
    increasing in sigma.
    """
    w = jnp.asarray(weights)
    return -jnp.sum(jnp.where(w > 0, w * jnp.log(jnp.where(w > 0, w, 1.0)), 0.0), axis=-1)


def figure2_setup(dtype=None):
    """The three signals and the query of Figure 2. Returns (data, y, mean)."""
    data = jnp.asarray(_FIGURE2_DATA, dtype=dtype)
    y = jnp.asarray(_FIGURE2_QUERY, dtype=dtype)
    return data, y, jnp.mean(data, axis=0)
