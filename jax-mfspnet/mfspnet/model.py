"""Dense blocks (Sec. 2.1, Eq. 1) and decayed weighted residual stacking
(Sec. 4.6, Eqs. 9-11, Alg. 4), in JAX.

A dense block is a run of composite layers, each BN -> ReLU -> 3x3 conv, and
each layer sees the concatenation of the block input and every earlier layer
output. With layer growth rates k_1..k_L the input width of layer l+1 is
Eq. 1, N_{l+1} = sum_{j=0..l} k_j, where k_0 is the block input width. Unlike
DenseNet the growth rate varies per layer, which is what the particle encodes.

Stacking follows Eq. 11:

    H_l = F_l(H_{l-1}) + sum_{i<l} alpha_i P_i(H_i),
    alpha_i = exp(-psi (l - i)) / sum_{k<l} exp(-psi (l - k)),

with psi = 0.88 and T_max = 8 (Table 2). P_i is a 1x1 projection that puts
every earlier block output on the width of H_{l-1} so the sum is well defined.
Each block ends in a 1x1 transition back to the stack width, which is what
keeps the FLOP curve flat in stack depth (Sec. 6.8).

Alg. 4 line 10 writes the aggregation with raw powers alpha^(k-i) and no
denominator, while Eqs. 10 and 11 normalize the weights to sum to one. The
normalized form is implemented, since the paper's own justification for it is
the normalization ("maintaining stable signal scaling across depths").

Parameters are plain nested dicts of arrays. There is no framework here beyond
jax and optax.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

PSI = 0.88
T_MAX = 8


def aggregation_weights(l: int, psi: float = PSI) -> np.ndarray:
    """Eq. 10: alpha_i for i = 1..l-1, normalized, nearest block heaviest."""
    if l < 2:
        return np.zeros(0)
    i = np.arange(1, l, dtype=np.float64)
    w = np.exp(-psi * (l - i))
    return w / w.sum()


def _conv_init(key, shape, fan_in):
    """He normal, the usual choice for ReLU stacks."""
    return jax.random.normal(key, shape) * math.sqrt(2.0 / fan_in)


def init_composite_layer(key, in_ch: int, growth: int):
    """BN scale/offset plus a 3x3 conv producing ``growth`` feature maps."""
    w = _conv_init(key, (3, 3, in_ch, growth), 9 * in_ch)
    return {
        "bn_scale": jnp.ones((in_ch,)),
        "bn_bias": jnp.zeros((in_ch,)),
        "conv": w,
    }


def init_dense_block(key, in_ch: int, growth_rates: Sequence[int], out_ch: int | None):
    """A block of composite layers, optionally followed by a 1x1 transition.

    ``out_ch`` is the width the block is compressed back to. Passing ``None``
    keeps the full DenseNet output, the concatenation of the block input and
    every layer output.
    """
    params = {"layers": []}
    ch = in_ch
    for g in growth_rates:
        key, sub = jax.random.split(key)
        params["layers"].append(init_composite_layer(sub, ch, int(g)))
        ch += int(g)  # Eq. 1
    if out_ch is not None:
        key, sub = jax.random.split(key)
        params["transition"] = {
            "bn_scale": jnp.ones((ch,)),
            "bn_bias": jnp.zeros((ch,)),
            "conv": _conv_init(sub, (1, 1, ch, out_ch), ch),
        }
    return params


def block_output_channels(in_ch: int, growth_rates: Sequence[int], out_ch: int | None) -> int:
    """Eq. 1 at the end of the block: k_0 + sum_l k_l, or the transition width."""
    if out_ch is not None:
        return int(out_ch)
    return int(in_ch) + int(sum(int(g) for g in growth_rates))


def _batch_norm(x, scale, bias, eps=1e-5):
    """BN over the batch and spatial axes, batch statistics.

    Only batch statistics are used. The toy runs here always evaluate on a full
    batch, so there are no running averages to carry, and the gradient study in
    the tests is a single forward and backward at initialization.
    """
    mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
    var = jnp.var(x, axis=(0, 1, 2), keepdims=True)
    xhat = (x - mean) * jax.lax.rsqrt(var + eps)
    return xhat * scale + bias


def _conv(x, w, stride=1):
    return jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def apply_dense_block(params, x):
    """Forward pass of one dense block.

    Feature maps are concatenated on the channel axis after every composite
    layer, so layer l reads everything before it (Sec. 2.1).
    """
    h = x
    for layer in params["layers"]:
        y = _batch_norm(h, layer["bn_scale"], layer["bn_bias"])
        y = jax.nn.relu(y)
        y = _conv(y, layer["conv"])
        h = jnp.concatenate([h, y], axis=-1)
    if "transition" in params:
        t = params["transition"]
        h = _batch_norm(h, t["bn_scale"], t["bn_bias"])
        h = jax.nn.relu(h)
        h = _conv(h, t["conv"])
    return h


@dataclasses.dataclass(frozen=True)
class StackConfig:
    """Shape of the network built around a decoded block."""

    n_classes: int
    stem_channels: int = 16
    n_blocks: int = 1
    psi: float = PSI
    dense_stacking: bool = True


def init_network(key, growth_rates: Sequence[int], in_channels: int, cfg: StackConfig):
    """Build parameters for stem -> n_blocks copies of the evolved block -> head.

    Every block is an independent copy with its own weights; the block
    *topology* is what the particle fixes, which is what gets stacked and
    transferred (Sec. 4.6). The stack width stays at ``stem_channels``, so the
    1x1 projections P_i are square and cheap.
    """
    width = cfg.stem_channels
    key, sub = jax.random.split(key)
    params = {
        "stem": _conv_init(sub, (3, 3, in_channels, width), 9 * in_channels),
        "blocks": [],
        "projections": [],
    }
    for l in range(cfg.n_blocks):
        key, sub = jax.random.split(key)
        params["blocks"].append(init_dense_block(sub, width, growth_rates, width))
        projections = []
        if cfg.dense_stacking:
            for _ in range(l):
                key, sub = jax.random.split(key)
                projections.append(_conv_init(sub, (1, 1, width, width), width))
        params["projections"].append(projections)
    key, sub = jax.random.split(key)
    params["head_w"] = jax.random.normal(sub, (width, cfg.n_classes)) * (1.0 / math.sqrt(width))
    params["head_b"] = jnp.zeros((cfg.n_classes,))
    return params


def apply_network(params, x, cfg: StackConfig):
    """Forward pass. Returns logits of shape (batch, n_classes)."""
    h = _conv(x, params["stem"])
    outputs = []
    for l, (block, projections) in enumerate(
        zip(params["blocks"], params["projections"]), start=1
    ):
        out = apply_dense_block(block, h)
        if cfg.dense_stacking and outputs:
            alpha = aggregation_weights(l, cfg.psi)
            agg = jnp.zeros_like(out)
            for a, p, prev in zip(alpha, projections, outputs):
                agg = agg + float(a) * _conv(prev, p)
            out = out + agg  # Eq. 11
        outputs.append(out)
        h = out
    h = jnp.mean(h, axis=(1, 2))
    return h @ params["head_w"] + params["head_b"]


def count_parameters(params) -> int:
    return int(sum(np.prod(np.shape(v)) for v in jax.tree.leaves(params)))
