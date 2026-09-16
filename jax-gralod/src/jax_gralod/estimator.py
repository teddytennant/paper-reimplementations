"""Stage-specific LOD estimator G_s, residual adaptor H_s, and probe P_s.

The paper leaves the internals of G_s / H_s / P_s underspecified beyond
"lightweight". Each is a small NHWC block: optional 1x1 projections, concat,
Conv3x3 -> SimpleGate -> Conv1x1.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from jax_gralod.ops import (
    Array,
    Params,
    conv1x1,
    conv3x3,
    init_conv2d,
    simple_gate,
)
from jax_gralod.query import clamp_lod, lod_query


def init_estimator(key: Array, dec_ch: int, lod_ch: int, n_levels: int) -> Params:
    """G_s: project D_s to C, concat {Z_s^l}, Conv3x3 -> SG -> Conv1x1 -> 1."""
    k1, k2, k3 = jax.random.split(key, 3)
    in_ch = (n_levels + 1) * lod_ch
    hidden = 2 * lod_ch
    return {
        "d_proj": init_conv2d(k1, 1, 1, dec_ch, lod_ch),
        "conv3": init_conv2d(k2, 3, 3, in_ch, hidden),
        "head": init_conv2d(k3, 1, 1, lod_ch, 1),
    }


def apply_estimator(
    params: Params,
    decoder_feat: Array,
    z_stage: Sequence[Array],
    n_levels: int,
) -> Array:
    """lambda_s(p) = G_s(D_s, {Z_s^l}) in [0, L-1] (Eq. 4)."""
    d_proj = conv1x1(params["d_proj"], decoder_feat)
    x = jnp.concatenate((d_proj,) + tuple(z_stage), axis=-1)
    x = conv3x3(params["conv3"], x)
    x = simple_gate(x)
    logits = conv1x1(params["head"], x)
    lam = jax.nn.sigmoid(logits) * float(n_levels - 1)
    return clamp_lod(lam, n_levels)


def init_adaptor(key: Array, dec_ch: int, lod_ch: int) -> Params:
    """H_s: project F_LOD to decoder channels, concat with D_s, Conv3x3 -> SG -> Conv1x1."""
    k1, k2, k3 = jax.random.split(key, 3)
    hidden = 2 * dec_ch
    return {
        "f_proj": init_conv2d(k1, 1, 1, lod_ch, dec_ch),
        "conv3": init_conv2d(k2, 3, 3, 2 * dec_ch, hidden),
        "head": init_conv2d(k3, 1, 1, dec_ch, dec_ch),
    }


def apply_adaptor(params: Params, decoder_feat: Array, f_lod: Array) -> Array:
    """Residual H_s(D_s, F_LOD^s). D_tilde = D_s + H_s(...) is applied by the caller."""
    f_proj = conv1x1(params["f_proj"], f_lod)
    x = jnp.concatenate([decoder_feat, f_proj], axis=-1)
    x = conv3x3(params["conv3"], x)
    x = simple_gate(x)
    return conv1x1(params["head"], x)


def residual_inject(decoder_feat: Array, residual: Array) -> Array:
    """Eq. 6 / 20: D_tilde_s = D_s + H_s(D_s, F_LOD^s)."""
    return decoder_feat + residual


def init_probe(key: Array, dec_ch: int, lod_ch: int, out_ch: int) -> Params:
    """P_s shared across levels within stage s. Concat(D_s_proj, Z_s^l) -> RGB/out_ch."""
    k1, k2, k3 = jax.random.split(key, 3)
    hidden = 2 * lod_ch
    return {
        "d_proj": init_conv2d(k1, 1, 1, dec_ch, lod_ch),
        "conv3": init_conv2d(k2, 3, 3, 2 * lod_ch, hidden),
        "head": init_conv2d(k3, 1, 1, lod_ch, out_ch),
    }


def apply_probe(
    params: Params,
    decoder_feat: Array,
    z_level: Array,
    detach_inputs: bool = True,
) -> Array:
    """xhat_s^l = P_s(Concat(D_s, Z_s^l)).

    Training uses stop-gradient on Concat(D_s, Z_s^l) (Eq. 29) so L_probe and
    MSFC errors do not hijack the backbone. Probes are discarded at inference.
    """
    d = decoder_feat
    z = z_level
    if detach_inputs:
        d = jax.lax.stop_gradient(d)
        z = jax.lax.stop_gradient(z)
    d_proj = conv1x1(params["d_proj"], d)
    x = jnp.concatenate([d_proj, z], axis=-1)
    x = conv3x3(params["conv3"], x)
    x = simple_gate(x)
    return conv1x1(params["head"], x)


def inject_stage(
    stage_params: Params,
    decoder_feat: Array,
    z_stage: Sequence[Array],
    n_levels: int,
) -> tuple[Array, Array, Array]:
    """Estimate lambda, query F_LOD, residual-inject. Returns (D_tilde, lambda, F_LOD)."""
    lam = apply_estimator(stage_params["G"], decoder_feat, z_stage, n_levels)
    f_lod = lod_query(lam, z_stage)
    residual = apply_adaptor(stage_params["H"], decoder_feat, f_lod)
    return residual_inject(decoder_feat, residual), lam, f_lod
