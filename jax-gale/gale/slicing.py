"""Sliced convolutional inference: the Local Exact branch of Section 3.

The input is cut into horizontal slices of full rows, P_W = W (Section 3.1,
Figure 2), not square patches. Each slice carries a halo of extra rows on both
sides. Classical PPBI sets that halo to R_0(L) once, at the input, and pays for
it at every depth. GaLe keeps a separate overlap O_t per tensor and lets the
calibration pass find the smallest one that meets an error target, so the deep
layers stop carrying a halo they no longer need (Section 3.1, Appendix A.3).

Row bookkeeping. Bands are defined on the last tensor of the stack and pushed
back up: a band [a, b) of the output maps to [a * S_t, b * S_t) on tensor t,
where S_t is the product of the strides below t. With zero halo the window at
every depth is exactly the band, and the pieces tile the output with no gap and
no overlap. A halo only ever grows the window, so the band stays inside it and
the final crop is well defined. Halos are rounded up to a multiple of S_t, which
keeps every window start divisible by the stride of the op consuming it; without
that the window's output grid is offset from the whole-map grid by a fraction of
a stride and nothing lines up.

Memory. ``MemoryTrace.arrays`` holds the element count of every window that got
built and ``buffers`` the full tensors a block reads from and writes to, so the
memory claims in the tests are measured rather than read off a formula. The
live-set model is crude and the same for both paths: an op has its input and its
output alive at once, so a whole-map pass peaks at max_t(size_t + size_{t+1}). A
sliced pass also holds the block's input and output buffers, since a tiling
runtime has to write its slices somewhere, so it peaks at those two plus the
largest window pair.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp

from .layers import stride_suffix, tensor_shapes


@dataclasses.dataclass
class MemoryTrace:
    """What a forward pass allocated: window sizes in ``arrays``, the largest
    input-plus-output pair in ``pair_peak``, whole tensors in ``buffers``."""

    arrays: list[int] = dataclasses.field(default_factory=list)
    pair_peak: int = 0
    buffers: int = 0

    @property
    def peak(self) -> int:
        return self.buffers + self.pair_peak

    def merge(self, other: "MemoryTrace") -> "MemoryTrace":
        return MemoryTrace(
            arrays=self.arrays + other.arrays,
            pair_peak=max(self.pair_peak, other.pair_peak),
            buffers=max(self.buffers, other.buffers),
        )


def heights(ops, shape):
    return [s[1] for s in tensor_shapes(ops, shape)]


def snap_halos(ops, halos):
    """Round each O_t up to a multiple of S_t so window starts stay aligned."""
    s = stride_suffix(ops)
    return [int(math.ceil(h / s[t]) * s[t]) for t, h in enumerate(halos)]


def overlap_profile(ops, shape, o_in, caps=None):
    """Per-tensor overlaps O_t implied by an input overlap O_0, Section 3.1.

    A convolution eats ``p = (k-1)//2`` rows off each end of a window and a
    stride divides what is left, so an overlap of O rows at tensor t-1 leaves
    ``(O - p) // s`` usable rows at tensor t. Propagating that down is the whole
    point of the method: PPBI carries R_0(L) rows at every layer, GaLe carries a
    profile that decays with depth, which is where the overhead drop from over
    100% to under 20% (Section 3) comes from.

    ``caps[t]`` is the largest overlap tensor t can hold inside the memory
    budget. A cap bites into everything below it, so it is applied during the
    propagation rather than after.
    """
    o = [0] * (len(ops) + 1)
    o[0] = int(o_in)
    for t, op in enumerate(ops):
        if caps is not None:
            o[t] = min(o[t], caps[t])
        o[t + 1] = max(0, (o[t] - (op.row_k - 1) // 2) // op.row_s)
    if caps is not None:
        o[-1] = min(o[-1], caps[-1])
    return snap_halos(ops, o)


def overlap_caps(ops, shape, n_slices, budget_elems):
    """Largest O_t at each tensor whose window still fits ``budget_elems``."""
    if budget_elems is None:
        return None
    shapes = tensor_shapes(ops, shape)
    bs = bands(ops, shape, n_slices)
    caps = []
    for t, (nn, h, w, c) in enumerate(shapes):
        band = max(b[t][1] - b[t][0] for b in bs)
        room = int(budget_elems) // max(1, nn * w * c) - band
        caps.append(max(0, min(h, room // 2)))
    return caps


def band_edges(h_out: int, n_slices: int) -> list[int]:
    n = min(n_slices, h_out)
    return [round(i * h_out / n) for i in range(n + 1)]


def bands(ops, shape, n_slices):
    """For each slice, the [lo, hi) row band it owns at every tensor depth."""
    h = heights(ops, shape)
    s = stride_suffix(ops)
    edges = band_edges(h[-1], n_slices)
    out = []
    for i in range(len(edges) - 1):
        out.append(
            [
                (edges[i] * s[t], min(h[t], edges[i + 1] * s[t]))
                for t in range(len(ops) + 1)
            ]
        )
    return out


def sliced_forward(ops, x, n_slices, halos=None):
    """Run ``ops`` slice by slice and merge, the L_E output of Section 3.1.

    ``halos[t]`` is the overlap O_t carried on tensor t, one number per tensor,
    in rows. Returns the merged output and a MemoryTrace of what was allocated.
    """
    shape = x.shape
    h = heights(ops, shape)
    hal = snap_halos(ops, [0] * (len(ops) + 1) if halos is None else list(halos))
    tr = MemoryTrace()
    pieces = []
    for band in bands(ops, shape, n_slices):
        lo = max(0, band[0][0] - hal[0])
        hi = min(h[0], band[0][1] + hal[0])
        cur = x[:, lo:hi]
        tr.arrays.append(cur.size)
        for t, op in enumerate(ops):
            if t > 0:
                nlo = max(lo, max(0, band[t][0] - hal[t]))
                nhi = min(hi, min(h[t], band[t][1] + hal[t]))
                if (nlo, nhi) != (lo, hi):
                    cur = cur[:, nlo - lo : nhi - lo]
                    lo, hi = nlo, nhi
                    tr.arrays.append(cur.size)
            in_size = cur.size
            cur, lo = op.apply_window(cur, lo, h[t])
            hi = lo + cur.shape[1]
            if hi > h[t + 1]:
                cur = cur[:, : h[t + 1] - lo]
                hi = h[t + 1]
            tr.arrays.append(cur.size)
            tr.pair_peak = max(tr.pair_peak, in_size + cur.size)
        blo, bhi = band[-1]
        assert lo <= blo and hi >= bhi, "window lost its own band"
        pieces.append(cur[:, blo - lo : bhi - lo])
    y = jnp.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
    tr.buffers = x.size + y.size
    return y, tr


def exact_forward_trace(ops, x):
    """Whole-map pass with the same live-set model, for the memory comparison."""
    from .layers import exact_forward_traced

    y, sizes = exact_forward_traced(ops, x)
    tr = MemoryTrace(arrays=list(sizes))
    tr.pair_peak = max(a + b for a, b in zip(sizes[:-1], sizes[1:]))
    return y, tr


@dataclasses.dataclass(frozen=True)
class BlockPlan:
    """Adaptive slicing, Section 3.2.

    ``starts`` splits the stack into blocks. Each block runs with its own slice
    count and the tensor between two blocks is materialized whole, which is the
    S-blocks-B configuration reported in Appendix E. Slice counts drop as the
    feature maps shrink, so deep blocks stop paying for overlap they no longer
    need.
    """

    starts: tuple[int, ...]
    n_slices: tuple[int, ...]
    halos: tuple[int, ...]

    def ranges(self, n_ops: int):
        ends = list(self.starts[1:]) + [n_ops]
        return list(zip(self.starts, ends))


def plan_adaptive(ops, shape, budget_elems, starts=(0,), halos=None):
    """Pick the smallest slice count per block that fits ``budget_elems``.

    Section 3.2: "during calibration, we compute the minimal number of patches
    required for each block to strictly satisfy the peak memory constraint. As
    feature map resolution decreases, GaLe progressively reduces the patch
    count".
    """
    shapes = tensor_shapes(ops, shape)
    sizes = [int(s[0] * s[1] * s[2] * s[3]) for s in shapes]
    h = heights(ops, shape)
    counts = []
    for a, b in BlockPlan(tuple(starts), (), ()).ranges(len(ops)):
        biggest = max(sizes[a : b + 1])
        n = max(1, -(-biggest // int(budget_elems)))
        counts.append(min(n, max(1, h[b])))
    halos = tuple([0] * (len(ops) + 1)) if halos is None else tuple(halos)
    return BlockPlan(tuple(starts), tuple(counts), halos)


def run_plan(ops, x, plan: BlockPlan):
    """Run a block plan end to end, materializing the tensor between blocks."""
    tr = MemoryTrace()
    cur = x
    for (a, b), n in zip(plan.ranges(len(ops)), plan.n_slices):
        cur, sub = sliced_forward(ops[a:b], cur, n, plan.halos[a : b + 1])
        tr = tr.merge(sub)
    return cur, tr
