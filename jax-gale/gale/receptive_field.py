"""Receptive field arithmetic, Equation 1 and the PPBI overlap it implies.

Equation 1 of the paper:

    R_0(L) = sum_{l=1..L} ( (k_l - 1) * prod_{i=1..l-1} s_i ) + 1

and Section 3.1 sets the classical partial patch based inference overlap to that
same number, O_ppbi = R_0(L): every patch has to carry the full receptive field
of the deepest tiled layer, on both sides, or its boundary outputs are wrong.

GaLe's claim is O_GaLe <= O_ppbi, with the gap paid for by an approximation
error the user caps at epsilon. The other half of the claim is that O_ppbi is
often not a finite number at all: any operator that reduces over the whole map
(squeeze-and-excite, spatial pyramid pooling, attention) has an unbounded
receptive field, which is why ``receptive_field`` returns ``math.inf`` when the
range contains one.
"""

from __future__ import annotations

import math


def receptive_field(ops, start: int = 0, end: int | None = None) -> float:
    """R_start(end) by Equation 1, measured in rows of tensor ``start``.

    ``ops[start:end]`` is the layer range. ``receptive_field(ops)`` is R_0(L),
    the input-side receptive field of the whole stack.
    """
    end = len(ops) if end is None else end
    total = 1
    stride_prod = 1
    for op in ops[start:end]:
        if op.global_ctx:
            return math.inf
        total += (op.row_k - 1) * stride_prod
        stride_prod *= op.row_s
    return total


def ppbi_overlap(ops) -> float:
    """O_ppbi = R_0(L), Section 3.1."""
    return receptive_field(ops)


def suffix_receptive_fields(ops) -> list[float]:
    """R_t(L) for every tensor t, the halo that makes tensor t's band exact.

    This is the setting the paper calls mathematically exact tiling, and it is
    the correctness anchor the sliced path is checked against.
    """
    return [receptive_field(ops, t) for t in range(len(ops) + 1)]


def calibration_step(ops, t: int) -> int:
    """Overlap increment for tensor t in Algorithm 1 (Appendix A.3).

    The pseudocode grows O_i by R_i(i+1) when layer i+1 is convolutional and by
    s_{i+1} otherwise. With halos indexed by tensor, tensor t is the input of
    ops[t], so R_t(t+1) is just that op's kernel extent.
    """
    if t >= len(ops):
        return 0
    op = ops[t]
    if op.is_conv:
        return max(1, op.row_k)
    return max(1, op.row_s)
