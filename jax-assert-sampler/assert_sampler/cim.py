"""The analog compute-in-memory noise model of arXiv:2609.00955, Section 3.1.

Equation 1 augments a matrix-vector multiply with a weight perturbation and an
output perturbation,

    y = x (W + eps_W) + eps_Y,   eps_W ~ N(0, sigma_W^2),  eps_Y ~ N(0, sigma_Y^2),

and Equation 2 makes the weight-noise variance depend on weight magnitude at a
fixed operating point,

    sigma_W^2 = alpha_W |W| + b_W.

The property the whole paper turns on is temporal: "Each mapped linear or
convolutional layer receives an independently sampled weight-perturbation map.
This map is fixed throughout one deployment realization and all of its denoising
steps, while eps_Y is redrawn for every MVM." So eps_W is spatial and persistent,
eps_Y is not, and prediction errors across denoising steps share one dominant
spatial pattern instead of averaging out.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CIMNoise(NamedTuple):
    """One operating point, the fitted triple (alpha_W, b_W, sigma_Y) of Section 3.1."""

    alpha_w: float = 0.0
    b_w: float = 0.0
    sigma_y: float = 0.0
    resample_per_step: bool = False
    """If True, eps_W is redrawn at every denoising step instead of held fixed.

    Physically this is not a chip; it is the control that separates "weight noise
    hurts" from "*persistent* weight noise hurts". The first-order argument in
    Section 3.2 needs a single Delta theta reused across steps for the cross terms
    of Equation 10 to survive, so with this on there is nothing left for the
    stochasticity to decorrelate.
    """

    @property
    def is_clean(self) -> bool:
        return self.alpha_w == 0.0 and self.b_w == 0.0 and self.sigma_y == 0.0


def weight_sigma(w: jnp.ndarray, noise: CIMNoise) -> jnp.ndarray:
    """sigma_W elementwise, Equation 2."""
    return jnp.sqrt(noise.alpha_w * jnp.abs(w) + noise.b_w)


def _weight_leaves(params):
    return [p["w"] for p in params]


def equivalent_level(params, noise: CIMNoise) -> float:
    """The paper's "equivalent weight-noise standard deviation": the RMS of
    sigma_W over every mapped weight. Section 4.1 reports noise strength this way,
    distinct from the sampler parameter eta.
    """
    sq = jnp.concatenate([weight_sigma(w, noise).ravel() ** 2
                          for w in _weight_leaves(params)])
    return float(jnp.sqrt(jnp.mean(sq)))


def calibrate(params, level: float, magnitude_fraction: float = 0.8,
              sigma_y: float = 0.0, resample_per_step: bool = False) -> CIMNoise:
    """Pick (alpha_W, b_W) so the equivalent weight-noise std equals ``level``.

    ``magnitude_fraction`` splits the total variance between the magnitude-dependent
    term alpha_W |W| and the constant floor b_W. The paper fits both from measured
    chip MVMs; here they are set so a requested noise level is reproducible across
    networks of different weight scale.
    """
    if not 0.0 <= magnitude_fraction <= 1.0:
        raise ValueError("magnitude_fraction must lie in [0, 1]")
    total_var = level ** 2
    mean_abs = float(jnp.mean(jnp.concatenate(
        [jnp.abs(w).ravel() for w in _weight_leaves(params)])))
    alpha_w = magnitude_fraction * total_var / mean_abs
    b_w = (1.0 - magnitude_fraction) * total_var
    return CIMNoise(alpha_w=alpha_w, b_w=b_w, sigma_y=sigma_y,
                    resample_per_step=resample_per_step)


def sample_mapping(key, params, noise: CIMNoise):
    """Draw one deployment realization of eps_W.

    Each layer gets an independently sampled map, and the result is meant to be
    drawn once and reused for every denoising step of that deployment. Repeated
    runs redraw it.
    """
    keys = jax.random.split(key, len(params))
    return [{"w": weight_sigma(p["w"], noise) * jax.random.normal(k, p["w"].shape)}
            for p, k in zip(params, keys)]


def zero_mapping(params):
    return [{"w": jnp.zeros_like(p["w"])} for p in params]


def perturb(params, mapping):
    """W -> W + eps_W. Biases stay digital, matching Section 4.1, where
    normalization, activations, residual adds and sampler control are digital."""
    return [{"w": p["w"] + m["w"], "b": p["b"]} for p, m in zip(params, mapping)]
