"""Eq. 2 / 17 aligner A_l and Eq. 3 / 19 resolution map R_{s,l}."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from jax_gralod.ops import (
    Array,
    Params,
    antialiased_resize,
    conv1x1,
    dwconv3x3,
    init_conv2d,
    init_dwconv3x3,
    init_layer_norm,
    layer_norm,
    sca,
    simple_gate,
)


def init_aligner(key: Array, in_ch: int, lod_ch: int) -> Params:
    """Lightweight A_l. First 1x1 maps Cl -> 2C so SimpleGate receives even channels."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    hidden = 2 * lod_ch
    return {
        "in_proj": init_conv2d(k1, 1, 1, in_ch, hidden),
        "dw": init_dwconv3x3(k2, hidden),
        "sca": init_conv2d(k3, 1, 1, lod_ch, lod_ch),
        "out_proj": init_conv2d(k4, 1, 1, lod_ch, lod_ch),
        "ln": init_layer_norm(lod_ch),
    }


def apply_aligner(params: Params, features: Array, ln_eps: float = 1e-5) -> Array:
    """Z^l = A_l(F^l).

    Prose order (Appendix A.1.2): Conv1x1 -> DWConv3x3 -> SimpleGate -> SCA
    -> Conv1x1 -> LayerNorm. The composition formula (Eq. 18) writes the
    operators right-to-left in a different order; this function follows the prose.
    """
    x = conv1x1(params["in_proj"], features)
    x = dwconv3x3(params["dw"], x)
    x = simple_gate(x)
    x = sca(params["sca"], x)
    x = conv1x1(params["out_proj"], x)
    return layer_norm(params["ln"], x, eps=ln_eps)


def resize_to_lattice(z: Array, height: int, width: int) -> Array:
    """R_{s,l}: identity / bilinear upsample / anti-aliased downsample (Eq. 19)."""
    return antialiased_resize(z, height, width)


def align_pyramid(
    align_params: Sequence[Params],
    encoder_features: Sequence[Array],
    ln_eps: float = 1e-5,
) -> tuple[Array, ...]:
    """Shared LOD hierarchy Z = {Z^0, ..., Z^{L-1}} (fine to coarse)."""
    if len(align_params) != len(encoder_features):
        raise ValueError("align_params and encoder_features must have the same length")
    return tuple(
        apply_aligner(p, f, ln_eps=ln_eps) for p, f in zip(align_params, encoder_features)
    )


def map_pyramid_to_stage(
    aligned: Sequence[Array],
    height: int,
    width: int,
) -> tuple[Array, ...]:
    """Z_s^l = R_{s,l}(Z^l) onto decoder stage-s lattice Omega_s."""
    return tuple(resize_to_lattice(z, height, width) for z in aligned)
