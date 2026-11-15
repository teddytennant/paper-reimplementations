"""Flow matching v-loss and the Focal Log-Frequency Loss (f-loss).

Paper: "Balancing Frequencies and Pixels in Flow Matching", arXiv:2609.02748.

Equation map
------------
Eq. (1)  L_v = E || v_theta - v* ||^2, the plain velocity regression.
Eq. (2)  L_v = E [ (1 / (1 - t)^2) || x_theta(x_t, t) - x_1 ||^2 ], the same
         objective rewritten as weighted x-regression. This is what ``v_loss``
         implements, and it is the "pixel" half of the joint objective.
Eq. (3)  L_f = sum_{u,v} (1 / (1 - t)^2) * (e_uv / max_{u'v'} e_u'v')
               * log(1 + |F_pred(u,v) - F_target(u,v)|),
         with e_uv = |F_pred(u,v) - F_target(u,v)|. Implemented by ``f_loss``.
         Three pieces, per the paper's own breakdown: the 1/(1-t)^2 timestep
         weight inherited unchanged from Eq. (2); the adaptive focal weight
         e_uv / max e, taken under a stop-gradient and described as a purely
         data-dependent coefficient that normalizes each sample; and the log
         compression log(1 + e_uv), which the paper reads as a continuous
         Laplacian pyramid, Eq. (5).
Eq. (4)  L = w_f * L_f + w_v * L_v with w_f = lambda(s), w_v = 1 - lambda(s).
         Implemented by ``fv_loss`` and ``flf.schedule``.
Eq. (5)  L_uv = log(1 + || F_theta(u,v) - F(u,v) ||_2). The norm is over the
         real and imaginary parts of the *complex difference*, so e_uv is the
         modulus of the residual spectrum, not the difference of moduli. Note
         that the paper's prose calls f-loss "insensitive to phase" while
         Eq. (3) and Eq. (5) as written do depend on phase; the equations are
         what is implemented here.

Generalized family
------------------
``f_loss`` exposes ``focal_exponent`` and ``spectral_term`` so the ablation of
Table 6 and the reduction to a plain spectral L2 both fall out of one function:

  focal_exponent=1, spectral_term="log"     Eq. (3), the paper's f-loss,
                                            best row of Table 6 (FID 41.78).
  focal_exponent=0, spectral_term="log"     the w_uv = 1 row of Table 6.
  focal_exponent=1, spectral_term="square"  the focal frequency loss of the
                                            prior work the paper compares
                                            against, Table 6 row 1.
  focal_exponent=0, spectral_term="square"  plain spectral L2, which by
                                            Parseval equals H * W * v_loss.

Reductions and hyperparameters
------------------------------
The timestep weight divides by (1 - t) clipped at 0.05, the value given for
"clip of (1 - t) in division" in Table 5.

Conventions: images are NHWC, the DFT is unnormalized, the spatial and channel
axes are summed and the batch axis is averaged. Summing matches the sum over
(u,v) in Eq. (3) and the squared norm in Eq. (2), and it is what makes the
Parseval constant exactly H * W.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .spectral import fft2

CLIP_1_MINUS_T = 0.05  # Table 5, "clip of (1 - t) in division"


def timestep_weight(t, clip: float = CLIP_1_MINUS_T):
    """1 / (1 - t)^2 with (1 - t) clipped below at ``clip``.

    Shared by Eq. (2) and Eq. (3); the paper keeps it unchanged in the
    frequency term so timestep dynamics stay consistent between the two.
    """
    t = jnp.asarray(t)
    return 1.0 / jnp.maximum(1.0 - t, clip) ** 2


def _per_sample_reduce(x):
    """Sum over spatial and channel axes, keep the batch axis."""
    return jnp.sum(x, axis=(-3, -2, -1))


def v_loss(pred, target, t, *, clip: float = CLIP_1_MINUS_T):
    """Pixel-space v-loss in x-prediction form, Eq. (2).

    ``pred`` is x_theta(x_t, t), ``target`` is x_1, ``t`` is one time per
    sample with shape (N,). Returns a scalar.
    """
    w = timestep_weight(t, clip)
    per_sample = _per_sample_reduce((pred - target) ** 2)
    return jnp.mean(w * per_sample)


def spectral_residual(pred, target):
    """|F_pred - F_target| per DFT bin, the e_uv of Eq. (3).

    The magnitude is computed from the squared modulus with a guard so the
    gradient at an exactly zero residual is 0 rather than NaN. Away from zero
    the value and the gradient are the ordinary ones.
    """
    diff = fft2(pred - target)
    sq = diff.real**2 + diff.imag**2
    safe = jnp.where(sq > 0, sq, 1.0)
    return jnp.where(sq > 0, jnp.sqrt(safe), jnp.zeros_like(sq)), sq


def f_loss(
    pred,
    target,
    t,
    *,
    focal_exponent: float = 1.0,
    spectral_term: str = "log",
    clip: float = CLIP_1_MINUS_T,
):
    """Focal Log-Frequency Loss, Eq. (3).

    The focal weight is (e_uv / max e)^focal_exponent under a stop-gradient,
    with the max taken over the spatial and channel axes of each sample, which
    is the paper's "normalizes each sample" reading of max_{u',v'}.

    ``spectral_term`` picks what the weight multiplies: "log" is log(1 + e_uv),
    Eq. (3) and Eq. (5); "linear" is e_uv, the uncompressed variant of the
    Table 6 ablation; "square" is e_uv^2, which turns the objective into a
    weighted spectral L2.

    ``focal_exponent=0`` makes the weight exactly 1 with no stop-gradient
    needed, so that path stays clean for the reduction test.
    """
    e, e_sq = spectral_residual(pred, target)

    if focal_exponent == 0.0:
        weight = jnp.ones_like(e)
    else:
        peak = jnp.max(e, axis=(-3, -2, -1), keepdims=True)
        peak = jnp.where(peak > 0, peak, 1.0)
        weight = jax.lax.stop_gradient((e / peak) ** focal_exponent)

    if spectral_term == "log":
        term = jnp.log1p(e)
    elif spectral_term == "linear":
        term = e
    elif spectral_term == "square":
        term = e_sq
    else:
        raise ValueError(f"unknown spectral_term: {spectral_term!r}")

    w = timestep_weight(t, clip)
    return jnp.mean(w * _per_sample_reduce(weight * term))


def fv_loss(pred, target, t, w_f, w_v, *, clip: float = CLIP_1_MINUS_T, **f_kwargs):
    """Joint objective of Eq. (4), L = w_f * L_f + w_v * L_v.

    ``flf.schedule.frequency_weight`` supplies w_f = lambda(s) and
    w_v = 1 - lambda(s).
    """
    return w_f * f_loss(pred, target, t, clip=clip, **f_kwargs) + w_v * v_loss(
        pred, target, t, clip=clip
    )
