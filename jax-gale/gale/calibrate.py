"""The calibration pass, Section 3.1 and Algorithm 1 of Appendix A.3.

Algorithm 1, transcribed:

    for i = 1..L:
        O_i <- 0;  N <- ceil(RAM_i / total RAM)
        repeat
          repeat
            run N-slice L_E inference with overlap O_i
            compute MSE_i
            O_i <- O_i + R_i(i+1) if layer i+1 is convolutional else O_i + s_{i+1}
          until MSE_i < eps or memory limit exceeded
          N <- N + 1
        until not memory limit exceeded

What is implemented here is that loop with the error read at the output of the
stack instead of at layer i, and with the per-layer overlaps tied together by
``overlap_profile`` instead of moved one at a time.

Both changes come from the same fact. The reconstruction error at the output
only moves when the whole overlap profile moves: an overlap at tensor t is
material only if the tensors above it hand down the rows it asks for, so a
search that grows O_1 alone against an output-level target just burns memory
until it saturates. Growing the input overlap and propagating it down gives a
profile that decays with depth, which is the one the paper describes, and turns
the search into a walk along a single ladder. The ladder rungs are Algorithm 1's
increment for the first layer, R_0(1) or s_1.

That has a second useful consequence. The sequence of configurations the search
visits does not depend on epsilon; only the stopping point does. So the total
overlap the pass returns is monotone non-increasing in epsilon, and never above
O_ppbi = R_0(L), the point past which extra rows change nothing.

The memory limit is the paper's. A per-tensor cap turns into a clip inside the
profile, and when the clipped profile stops improving, N goes up so a shorter
band leaves room for a wider halo.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .fusion import gale_forward
from .layers import exact_forward
from .receptive_field import calibration_step, ppbi_overlap, receptive_field
from .slicing import overlap_caps, overlap_profile


@dataclasses.dataclass(frozen=True)
class Calibration:
    """What the pass decided: the per-tensor overlaps and what they cost."""

    halos: tuple[int, ...]
    n_slices: int
    input_overlap: int
    mse: float
    met: bool
    passes: int

    @property
    def total_overlap(self) -> int:
        return int(sum(self.halos))


def calibrate(
    ops,
    x,
    epsilon: float,
    n_slices: int = 2,
    budget_elems=None,
    ga_factor: int | None = None,
    alpha: float = 1.0,
    max_slices: int | None = None,
    reference=None,
):
    """Grow the overlap until reconstruction MSE drops below epsilon.

    Returns a Calibration whose ``met`` flag says whether the target was reached
    inside the memory limit. ``halos`` is O_t for every tensor, ready to hand to
    ``sliced_forward``.
    """
    ref = exact_forward(ops, x) if reference is None else reference
    shape = x.shape
    max_slices = n_slices if max_slices is None else max_slices
    step = max(1, calibration_step(ops, 0))
    ceiling = receptive_field(ops)
    ceiling = int(ceiling) if ceiling != float("inf") else int(shape[1])
    passes = 0

    def run(halos, n):
        nonlocal passes
        passes += 1
        y, _ = gale_forward(ops, x, n, halos, ga_factor=ga_factor, alpha=alpha)
        return float(jnp.mean((y - ref) ** 2))

    worst = None
    for n in range(n_slices, max_slices + 1):
        caps = overlap_caps(ops, shape, n, budget_elems)
        seen = set()
        o_in = 0
        while True:
            halos = overlap_profile(ops, shape, o_in, caps)
            err = run(halos, n)
            if err < epsilon:
                return Calibration(tuple(halos), n, o_in, err, True, passes)
            key = tuple(halos)
            if key in seen and o_in > 0:
                break  # the memory cap is holding the profile down
            seen.add(key)
            if o_in >= ceiling:
                break
            o_in += step
        worst = Calibration(tuple(halos), n, o_in, err, False, passes)
    return worst


def overlap_vs_ppbi(ops, cal: Calibration) -> tuple[float, float]:
    """(O_GaLe, O_ppbi) at the input tensor, the comparison of Section 3.1."""
    return float(cal.halos[0]), ppbi_overlap(ops)
