"""Coordinate Decoder: fuse query, LCA, classifier.

Paper:

    f_p = Psi(P_{x,y})
    f_c = Omega(x, y)
    f_q = phi([f_p, f_c])
    f_q' = CA(f_q, F_local)
    z_hat = SH(f_q')

Sparse decode queries an arbitrary list of ``(x, y)`` and returns
``(N, K)`` logits without allocating an ``H x W`` label map. Dense decode
queries a full coordinate grid and reshapes to ``(H, W, K)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from coord_lca.coords import (
    dense_query_coords,
    image_to_encoder_grid,
    normalize_coords,
    fourier_embed,
)
from coord_lca.lca import init_lca, localized_cross_attention
from coord_lca.patch import (
    embed_patches,
    extract_patches,
    init_linear,
    init_mlp,
    linear,
    mlp,
)


def init_decoder(
    key: jax.Array,
    *,
    d_model: int = 32,
    d_patch: int = 32,
    n_freq: int = 6,
    n_classes: int = 2,
    patch_size: int = 8,
    n_heads: int = 4,
    in_channels: int = 3,
) -> dict:
    """Initialize Coordinate Decoder parameters (arrays only)."""
    k_patch, k_fuse, k_lca, k_head = jax.random.split(key, 4)
    in_dim = patch_size * patch_size * in_channels
    d_coord = 4 * n_freq
    hidden = max(d_patch, in_dim // 2)
    return {
        "patch": init_mlp(k_patch, (in_dim, hidden, d_patch)),
        "fuse": init_mlp(k_fuse, (d_patch + d_coord, d_model, d_model)),
        "lca": init_lca(k_lca, d_model=d_model, n_heads=n_heads),
        "head": init_linear(k_head, d_model, n_classes),
    }


def fuse_query(fuse_params: list, patch_feat: jnp.ndarray, coord_feat: jnp.ndarray) -> jnp.ndarray:
    """``f_q = phi([f_p, f_c])`` with linear-ReLU layers, output dim ``d_model``."""
    x = jnp.concatenate([patch_feat, coord_feat], axis=-1)
    return mlp(fuse_params, x)


def build_queries(
    decoder_params: dict,
    image: jnp.ndarray,
    xy: jnp.ndarray,
    *,
    patch_size: int = 8,
    n_freq: int = 6,
) -> jnp.ndarray:
    """Coordinate-aware queries ``f_q`` of shape ``(N, d_model)``."""
    height, width = image.shape[:2]
    patches = extract_patches(image, xy, patch_size=patch_size)
    f_p = embed_patches(decoder_params["patch"], patches)
    f_c = fourier_embed(normalize_coords(xy, height, width), n_freq=n_freq)
    return fuse_query(decoder_params["fuse"], f_p, f_c)


def decode_sparse(
    decoder_params: dict,
    image: jnp.ndarray,
    tokens: jnp.ndarray,
    xy: jnp.ndarray,
    *,
    patch_size: int = 8,
    n_freq: int = 6,
    downsample: int = 2,
    encoder_patch_size: int = 4,
    include_global: bool = True,
) -> jnp.ndarray:
    """Predict class logits at arbitrary coordinates.

    Parameters
    ----------
    image:
        ``(H, W, C)`` full-resolution image (local patches only).
    tokens:
        Encoder grid ``(H_e, W_e, D)``.
    xy:
        ``(N, 2)`` pixel coordinates. Output is ``(N, K)`` — never an
        ``H x W`` map.

    Returns
    -------
    logits of shape ``(N, n_classes)``.
    """
    xy = jnp.asarray(xy, dtype=jnp.float32)
    if xy.ndim == 1:
        xy = xy[None, :]
    height, width = image.shape[:2]
    queries = build_queries(
        decoder_params, image, xy, patch_size=patch_size, n_freq=n_freq
    )
    xy_enc = image_to_encoder_grid(
        xy,
        height=height,
        width=width,
        downsample=downsample,
        encoder_patch_size=encoder_patch_size,
    )
    refined = localized_cross_attention(
        decoder_params["lca"],
        queries,
        tokens,
        xy_enc,
        include_global=include_global,
    )
    return linear(decoder_params["head"], refined)


def decode_dense(
    decoder_params: dict,
    image: jnp.ndarray,
    tokens: jnp.ndarray,
    *,
    patch_size: int = 8,
    n_freq: int = 6,
    downsample: int = 2,
    encoder_patch_size: int = 4,
    include_global: bool = True,
    centers: bool = True,
) -> jnp.ndarray:
    """Dense mask by querying a full coordinate grid. Shape ``(H, W, K)``."""
    height, width = image.shape[:2]
    xy = dense_query_coords(height, width, centers=centers).reshape(-1, 2)
    logits = decode_sparse(
        decoder_params,
        image,
        tokens,
        xy,
        patch_size=patch_size,
        n_freq=n_freq,
        downsample=downsample,
        encoder_patch_size=encoder_patch_size,
        include_global=include_global,
    )
    n_classes = logits.shape[-1]
    return logits.reshape(height, width, n_classes)


def cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Mean multi-class cross-entropy. ``logits (N, K)``, ``labels (N,)`` int."""
    labels = labels.astype(jnp.int32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    n = labels.shape[0]
    return -jnp.mean(log_probs[jnp.arange(n), labels])


def soft_dice(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    *,
    eps: float = 0.0,
) -> jnp.ndarray:
    """Soft Dice coefficient on flattened ``pred`` and ``target`` in ``[0, 1]``.

    With ``eps=0`` and a non-empty binary mask, ``dice(y, y) = 1`` and
    ``dice(y, 1 - y) = 0``.
    """
    pred = jnp.asarray(pred, dtype=jnp.float32).reshape(-1)
    target = jnp.asarray(target, dtype=jnp.float32).reshape(-1)
    intersection = jnp.sum(pred * target)
    denom = jnp.sum(pred) + jnp.sum(target)
    return jnp.where(denom > 0, (2.0 * intersection + eps) / (denom + eps), 1.0)


def binary_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean BCE with logits."""
    logits = jnp.asarray(logits, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    max_val = jnp.maximum(logits, 0.0)
    loss = max_val - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return jnp.mean(loss)


def coord_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    lambda_ce: float = 1.0,
    lambda_dice: float = 1.0,
    lambda_bce: float = 1.0,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Paper-style mix: multi-class CE plus per-class Dice and BCE (eq. 10)."""
    n_classes = logits.shape[-1]
    ce = cross_entropy(logits, labels)
    one_hot = jax.nn.one_hot(labels.astype(jnp.int32), n_classes)
    # Per-class binary scores from independent logits (paper uses class maps).
    dice_k = jax.vmap(
        lambda logit_k, tgt_k: 1.0
        - soft_dice(jax.nn.sigmoid(logit_k), tgt_k, eps=eps),
        in_axes=(1, 1),
    )(logits, one_hot)
    bce_k = jax.vmap(binary_cross_entropy, in_axes=(1, 1))(logits, one_hot)
    return lambda_ce * ce + (lambda_dice * jnp.mean(dice_k) + lambda_bce * jnp.mean(bce_k))
