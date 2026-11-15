"""The Global Approximate branch and the alpha fusion, Section 3.3 and Appendix C.

The L_E slices are spatially isolated, so they have no idea what the rest of the
map looks like. G_A fixes that by running the same ops on a bilinear downsample
of the whole input, then upsampling the result and mixing it in:

    out = alpha * L_E + (1 - alpha) * up(G_A)

Section 3.3 picks bilinear downsampling as the best quality/cost trade-off among
the compression functions the authors tried, and merges with linear
interpolation. Appendix C sweeps alpha on ImageNet: the curve is flat over
[0.7, 0.9] and peaks around 0.8, above both alpha = 1 (pure slicing, no global
context) and alpha = 0.5 (too much weight on a map that lost its high
frequencies). ALPHA_DEFAULT is that 0.8.

The downsample factor is the paper's N, chosen during calibration so the reduced
tensor fits the memory budget.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .layers import exact_forward_traced
from .slicing import MemoryTrace, sliced_forward

ALPHA_DEFAULT = 0.8
ALPHA_STABLE_RANGE = (0.7, 0.9)


def bilinear_down(x, factor: int):
    n, h, w, c = x.shape
    return jax.image.resize(
        x, (n, max(1, -(-h // factor)), max(1, -(-w // factor)), c), method="bilinear"
    )


def bilinear_up(x, hw):
    n, _, _, c = x.shape
    return jax.image.resize(x, (n, hw[0], hw[1], c), method="bilinear")


def ga_branch(ops, x, factor: int, out_hw=None):
    """Downsample by N, run the same ops, upsample back. Returns output and trace."""
    xs = bilinear_down(x, factor)
    y, sizes = exact_forward_traced(ops, xs)
    tr = MemoryTrace(arrays=[xs.size] + list(sizes))
    tr.pair_peak = max(a + b for a, b in zip(sizes[:-1], sizes[1:]))
    tr.buffers = xs.size
    if out_hw is not None and (y.shape[1], y.shape[2]) != tuple(out_hw):
        y = bilinear_up(y, out_hw)
        tr.arrays.append(y.size)
    return y, tr


def fuse(le, ga, alpha: float = ALPHA_DEFAULT):
    """Linear interpolation between the two branches, Section 3.3."""
    return alpha * le + (1.0 - alpha) * ga


def gale_forward(
    ops,
    x,
    n_slices: int,
    halos=None,
    ga_factor: int | None = 2,
    alpha: float = ALPHA_DEFAULT,
):
    """Full GaLe conv inference: sliced L_E, downsampled G_A, alpha fusion.

    ``ga_factor=None`` or ``alpha=1`` skips the G_A branch, which is the L_E only
    row of Table 1 and the configuration Appendix E labels "-Le".
    """
    le, tr = sliced_forward(ops, x, n_slices, halos)
    if ga_factor is None or alpha >= 1.0:
        return le, tr
    ga, ga_tr = ga_branch(ops, x, ga_factor, out_hw=(le.shape[1], le.shape[2]))
    out = fuse(le, ga, alpha)
    tr = tr.merge(ga_tr)
    tr.arrays.append(out.size)
    return out, tr


def mse(a, b) -> float:
    return float(jnp.mean((a - b) ** 2))
