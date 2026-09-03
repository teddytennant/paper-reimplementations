"""Layer primitives with an explicit row-window interface.

Everything here works on NHWC tensors, the layout the paper assumes for embedded
runtimes (Section 3.1: "when working with tensors in NHWC format ... slices align
naturally with memory layout"). Slicing is done over full rows, P_W = W, so the
width axis is always processed whole and only the H axis carries a window.

Each op exposes two things the rest of the package needs:

* ``row_k`` / ``row_s``, the kernel extent and stride along H. These feed the
  receptive field arithmetic of Equation 1.
* ``apply_window(x, lo, h_in)``, which runs the op on a contiguous band of rows
  ``[lo, lo + x.shape[1])`` taken from a tensor of full height ``h_in``.

``apply_window`` treats the band as a standalone tensor: it pads by ``(k-1)//2``
on both sides no matter where the band sits. That is what a tiling runtime does,
and it is the source of the approximation the paper controls with the overlap.
Rows near a band edge that has real neighbouring data are wrong; rows further in
than the receptive field are bit-exact. Running with ``lo=0`` and the whole
tensor recovers ordinary SAME-padded inference, so the exact path and the sliced
path go through the same code.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax import lax

HIGHEST = lax.Precision.HIGHEST


def same_out(h: int, s: int) -> int:
    """Output height of a SAME-padded op with stride ``s``."""
    return -(-h // s)


@dataclasses.dataclass(frozen=True)
class Conv:
    """Odd-kernel SAME-padded convolution, stride shared by H and W.

    ``w`` has shape (k, k, cin, cout). Matmuls run at HIGHEST precision so the
    exactness checks are not silently degraded to TF32 on NVIDIA.
    """

    w: jax.Array
    b: jax.Array
    stride: int = 1
    is_conv: bool = True
    global_ctx: bool = False

    @property
    def row_k(self) -> int:
        return int(self.w.shape[0])

    @property
    def row_s(self) -> int:
        return self.stride

    def out_shape(self, h: int, w: int, c: int) -> tuple[int, int, int]:
        return same_out(h, self.stride), same_out(w, self.stride), int(self.w.shape[3])

    def apply_window(self, x, lo, h_in):
        k, s = self.row_k, self.stride
        p = (k - 1) // 2
        xp = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        y = lax.conv_general_dilated(
            xp,
            self.w.astype(x.dtype),
            (s, s),
            "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            precision=HIGHEST,
        )
        return y + self.b.astype(x.dtype), lo // s


@dataclasses.dataclass(frozen=True)
class MaxPool:
    """SAME-padded max pool. Not a convolution, so calibration grows the overlap
    around it by ``s`` rather than by a kernel receptive field (Appendix A.3)."""

    k: int
    stride: int = 2
    is_conv: bool = False
    global_ctx: bool = False

    @property
    def row_k(self) -> int:
        return self.k

    @property
    def row_s(self) -> int:
        return self.stride

    def out_shape(self, h, w, c):
        return same_out(h, self.stride), same_out(w, self.stride), c

    def apply_window(self, x, lo, h_in):
        k, s = self.k, self.stride
        p = (k - 1) // 2
        neg = jnp.asarray(jnp.finfo(x.dtype).min, x.dtype)
        xp = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), constant_values=neg)
        y = lax.reduce_window(xp, neg, lax.max, (1, k, k, 1), (1, s, s, 1), "VALID")
        return y, lo // s


@dataclasses.dataclass(frozen=True)
class Relu:
    is_conv: bool = False
    global_ctx: bool = False
    row_k: int = 1
    row_s: int = 1

    def out_shape(self, h, w, c):
        return h, w, c

    def apply_window(self, x, lo, h_in):
        return jnp.maximum(x, 0), lo


@dataclasses.dataclass(frozen=True)
class GlobalGate:
    """Squeeze-and-excite style gate: ``x * sigmoid(gain * mean_hw(x) + bias)``.

    This is the operator Section 3.3 and Table 2 use to make the point that PPBI
    is not just expensive here but inapplicable: the receptive field covers the
    whole map, so no finite overlap makes a slice exact. A sliced pass reduces
    over the rows it happens to hold, and the G_A branch is what puts the global
    statistic back.
    """

    gain: jax.Array
    bias: jax.Array
    is_conv: bool = False
    global_ctx: bool = True
    row_k: int = 1
    row_s: int = 1

    def out_shape(self, h, w, c):
        return h, w, c

    def apply_window(self, x, lo, h_in):
        m = jnp.mean(x, axis=(1, 2), keepdims=True)
        return x * jax.nn.sigmoid(self.gain.astype(x.dtype) * m + self.bias.astype(x.dtype)), lo


def tensor_shapes(ops, shape):
    """Shapes of every tensor in the stack, input first, one per op after that."""
    n, h, w, c = shape
    out = [(n, h, w, c)]
    for op in ops:
        h, w, c = op.out_shape(h, w, c)
        out.append((n, h, w, c))
    return out


def exact_forward(ops, x):
    """Whole-map inference, the reference the calibration pass measures against."""
    h = x.shape[1]
    y = x
    for op in ops:
        y, _ = op.apply_window(y, 0, h)
        h = y.shape[1]
    return y


def exact_forward_traced(ops, x):
    """Whole-map inference that also returns the size of every array it built."""
    sizes = [x.size]
    h = x.shape[1]
    y = x
    for op in ops:
        y, _ = op.apply_window(y, 0, h)
        h = y.shape[1]
        sizes.append(y.size)
    return y, sizes


def stride_suffix(ops):
    """``S[t] = prod(row_s of ops[t:])``, the factor mapping a row of tensor t to
    rows of the final tensor. Used to place a band consistently at every depth."""
    s = [1] * (len(ops) + 1)
    for t in range(len(ops) - 1, -1, -1):
        s[t] = s[t + 1] * ops[t].row_s
    return s


def randn(key, shape, dtype):
    return jax.random.normal(key, shape, dtype=dtype)


def make_stack(key, cin, widths, kernels, strides, gate_at=None, dtype=jnp.float64):
    """Small conv stack for tests and demos: conv, relu, conv, relu, ...

    ``gate_at`` inserts a GlobalGate after that op index, giving a stack PPBI
    cannot tile.
    """
    ops = []
    c = cin
    for i, (co, k, s) in enumerate(zip(widths, kernels, strides)):
        key, k1, k2 = jax.random.split(key, 3)
        scale = 1.0 / math.sqrt(k * k * c)
        ops.append(
            Conv(
                w=randn(k1, (k, k, c, co), dtype) * scale,
                b=randn(k2, (co,), dtype) * 0.1,
                stride=s,
            )
        )
        ops.append(Relu())
        c = co
        if gate_at is not None and i == gate_at:
            key, k3 = jax.random.split(key)
            ops.append(GlobalGate(gain=jnp.ones((c,), dtype) * 2.0, bias=randn(k3, (c,), dtype)))
    return ops
