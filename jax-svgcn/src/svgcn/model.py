"""SMEF (eq. 4–6) + VF-STGCN (eq. 7–12). Smaller backbone than the paper."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from svgcn.features import build_features
from svgcn.layers import (
    conv_temporal,
    downsample_mask,
    layer_norm_channels,
    linear_nctj,
    make_mask,
    mask_4d,
    masked_gap,
    masked_group_norm,
    spatial_gcn,
)
from svgcn.skeleton import normalized_adjacency

NUM_CLASSES = 4
# Paper slices f_rel[0:7] / [8:9]; read inclusive as 8 linear + 2 angular (10-D).
COORD_SLICE = slice(0, 3)
LINEAR_SLICE = slice(3, 11)
ANGULAR_SLICE = slice(11, 14)


def _split(key, n=2):
    return jax.random.split(key, n)


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jnp.ndarray]:
    w = jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * jnp.sqrt(2.0 / in_dim)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": w, "b": b}


def _init_ln(dim: int) -> dict[str, jnp.ndarray]:
    return {
        "scale": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _init_conv(key: jax.Array, out_c: int, in_c: int, k: int) -> dict[str, jnp.ndarray]:
    w = jax.random.normal(key, (out_c, in_c, k, 1), dtype=jnp.float32) * jnp.sqrt(
        2.0 / (in_c * k)
    )
    b = jnp.zeros((out_c,), dtype=jnp.float32)
    return {"w": w, "b": b}


def _init_proj(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, Any]:
    return {"ln": _init_ln(in_dim), "lin": _init_linear(key, in_dim, out_dim)}


def _apply_proj(x: jnp.ndarray, p: dict[str, Any]) -> jnp.ndarray:
    x = layer_norm_channels(x, p["ln"]["scale"], p["ln"]["bias"])
    return jax.nn.relu(linear_nctj(x, p["lin"]["w"], p["lin"]["b"]))


def init_smef(key: jax.Array, embed_dim: int = 8, attn_hidden: int | None = None) -> dict[str, Any]:
    k1, k2, k3, k4, k5 = _split(key, 5)
    c = 3 * embed_dim
    if attn_hidden is None:
        attn_hidden = max(c // 4, 4)
    return {
        "coord": _init_proj(k1, 3, embed_dim),
        "linear": _init_proj(k2, 8, embed_dim),
        "angular": _init_proj(k3, 3, embed_dim),
        "embed_ln": _init_ln(c),
        "attn1": _init_linear(k4, c, attn_hidden),
        "attn2": _init_linear(k5, attn_hidden, c),
    }


def apply_smef(params: dict[str, Any], x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Eq. 4–6. ``x`` is ``(N, 14, T, J)``, ``mask`` is ``(N, T)``."""
    coord = _apply_proj(x[:, COORD_SLICE], params["coord"])
    linear = _apply_proj(x[:, LINEAR_SLICE], params["linear"])
    angular = _apply_proj(x[:, ANGULAR_SLICE], params["angular"])
    embed = jnp.concatenate([coord, linear, angular], axis=1)
    embed_n = layer_norm_channels(embed, params["embed_ln"]["scale"], params["embed_ln"]["bias"])
    pooled = masked_gap(embed_n, mask)
    h = jax.nn.relu(pooled @ params["attn1"]["w"] + params["attn1"]["b"])
    w_c = jax.nn.sigmoid(h @ params["attn2"]["w"] + params["attn2"]["b"])
    attn = embed * w_c[:, :, None, None] * mask_4d(mask)
    return attn


def _init_stgcn_block(
    key: jax.Array,
    in_c: int,
    out_c: int,
    k_t: int,
    stride: int,
    num_groups: int,
) -> dict[str, Any]:
    k1, k2, k3 = _split(key, 3)
    residual = None
    if in_c != out_c or stride != 1:
        residual = _init_conv(k3, out_c, in_c, 1)
    # Hyperparameters stay Python-side so the pytree is all float arrays.
    return {
        "gn": _init_ln(in_c),
        "gcn": _init_linear(k1, in_c, out_c),
        "tcn": _init_conv(k2, out_c, out_c, k_t),
        "residual": residual,
    }


def apply_stgcn_block(
    params: dict[str, Any],
    x: jnp.ndarray,
    mask: jnp.ndarray,
    adj: jnp.ndarray,
    num_groups: int = 4,
    stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h = masked_group_norm(x, mask, params["gn"]["scale"], params["gn"]["bias"], num_groups)
    h = h * mask_4d(mask)
    h = jax.nn.relu(spatial_gcn(h, adj, params["gcn"]["w"], params["gcn"]["b"]))
    h = h * mask_4d(mask)
    h = conv_temporal(h, params["tcn"]["w"], params["tcn"]["b"], mask=mask, stride=stride)
    if params["residual"] is None:
        res = x
    else:
        res = conv_temporal(
            x, params["residual"]["w"], params["residual"]["b"], mask=mask, stride=stride
        )
    lengths = jnp.sum(mask, axis=1).astype(jnp.int32)
    new_mask, _ = downsample_mask(lengths, x.shape[2], stride)
    t_out = h.shape[2]
    new_mask = new_mask[:, :t_out]
    out = jax.nn.relu(h + res) * mask_4d(new_mask)
    return out, new_mask, jnp.sum(new_mask, axis=1)


def init_svgcn(
    key: jax.Array,
    num_classes: int = NUM_CLASSES,
    embed_dim: int = 8,
    channels: tuple[int, ...] = (16, 32),
    blocks_per_group: int = 2,
    num_groups: int = 4,
    kernels: tuple[int, ...] = (1, 3),
) -> dict[str, Any]:
    """Smaller than the paper's 64/128/256 × 3×3 blocks; same module types."""
    k_smef, key = _split(key)
    smef = init_smef(k_smef, embed_dim=embed_dim)
    in_c = 3 * embed_dim
    groups = []
    for gi, out_c in enumerate(channels):
        k_t = kernels[min(gi, len(kernels) - 1)]
        blocks = []
        for bi in range(blocks_per_group):
            key, kb = _split(key)
            # Stride 1 keeps padded vs truncated logits comparable (eq. 10).
            # The paper uses s=2 between groups; this smaller backbone does not.
            stride = 1
            blocks.append(
                _init_stgcn_block(kb, in_c, out_c, k_t, stride, num_groups)
            )
            in_c = out_c
        groups.append(blocks)
    key, kf = _split(key)
    fc = _init_linear(kf, in_c, num_classes)
    return {
        "smef": smef,
        "groups": groups,
        "fc": fc,
        "adj": normalized_adjacency(),
    }


def apply_svgcn(
    params: dict[str, Any],
    coords: jnp.ndarray,
    lengths: jnp.ndarray,
) -> jnp.ndarray:
    """Logits ``(N, num_classes)``. ``coords`` is ``(N, T, J, 3)``."""
    t_max = coords.shape[1]
    mask = make_mask(lengths, t_max)
    feats = build_features(coords) * mask_4d(mask)
    x = apply_smef(params["smef"], feats, mask)
    adj = jax.lax.stop_gradient(params["adj"])
    for group in params["groups"]:
        for block in group:
            x, mask, _ = apply_stgcn_block(block, x, mask, adj, num_groups=4, stride=1)
    pooled = masked_gap(x, mask)
    return pooled @ params["fc"]["w"] + params["fc"]["b"]


def cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Mean softmax CE (eq. 11–12). ``labels`` are integer class ids."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    n = labels.shape[0]
    return -jnp.mean(logp[jnp.arange(n), labels])
