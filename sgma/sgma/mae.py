"""Tiny masked autoencoder stand-in.

Not the paper's ViT. Visible tokens are resized to a fixed grid, linearly
embedded to d=32, and passed through one pre-norm transformer block. The
decoder writes a learned mask token at masked positions, adds the mean
encoder state so masked predictions depend on visible tokens, adds index
embeddings, and applies one linear layer to patch pixels. Loss is MSE on
masked patches only.

``train_step`` returns ``(new_params, loss)`` after one SGD update. Params
are not mutated.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from sgma.damped import canvas
from sgma.masking import structure_mask
from sgma.tokenizer import tokenize_with_depths

D_MODEL = 32
PATCH_SIZE = 4
DEFAULT_TARGET_N = 13
DEFAULT_RATIO = 0.75
N_HEADS = 2
MLP_RATIO = 4


def key_to_seed(key):
    if isinstance(key, (int, np.integer)):
        return int(key)
    data = np.asarray(jax.random.key_data(key)).astype(np.uint32).ravel()
    return int(data[0])


def _ensure_nhwc(images):
    images = jnp.asarray(images)
    if images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4:
        raise ValueError(f"images must be (B, H, W) or (B, H, W, C), got {images.shape}")
    return images.astype(jnp.float32)


def _structure_map(salience):
    salience = np.asarray(salience, dtype=np.float64)
    if salience.ndim == 3:
        salience = salience[0]
    if salience.ndim != 2:
        raise ValueError(f"salience must be (H, W) or (B, H, W), got {salience.shape}")
    return salience


def prepare(salience, key, target_n=DEFAULT_TARGET_N, ratio=DEFAULT_RATIO):
    """Deterministic tokens plus a structure mask. Concrete, not differentiated."""
    structure = _structure_map(salience)
    tokens, depths = tokenize_with_depths(structure, target_n, deterministic=True)
    variances = canvas(structure, tokens, depths)
    mask = structure_mask(variances, ratio=ratio, seed=key_to_seed(key))
    return tokens, depths, mask, variances


def _to_patch(crop, out):
    """Differentiable resize of (B, h, w, C) onto an ``out`` x ``out`` grid."""
    _batch, height, width, _channels = crop.shape
    if height == out and width == out:
        return crop
    if height >= out and width >= out and height % out == 0 and width % out == 0:
        block_h = height // out
        block_w = width // out
        reshaped = crop.reshape(crop.shape[0], out, block_h, out, block_w, crop.shape[-1])
        return reshaped.mean(axis=(2, 4))
    if height <= out and width <= out and out % height == 0 and out % width == 0:
        return jnp.repeat(jnp.repeat(crop, out // height, axis=1), out // width, axis=2)
    return jax.image.resize(
        crop, (crop.shape[0], out, out, crop.shape[-1]), method="linear"
    )


def extract_patches(images, tokens, patch_size=PATCH_SIZE):
    images = _ensure_nhwc(images)
    flats = []
    for r0, r1, c0, c1 in tokens:
        crop = images[:, r0:r1, c0:c1, :]
        resized = _to_patch(crop, patch_size)
        flats.append(jnp.reshape(resized, (images.shape[0], -1)))
    return jnp.stack(flats, axis=1)


def init_params(key, channels=1, patch_size=PATCH_SIZE, d=D_MODEL, max_tokens=64):
    """Learned embed, index table, one encoder block, mask token, linear decoder."""
    key = jax.random.PRNGKey(key) if isinstance(key, (int, np.integer)) else key
    patch_dim = patch_size * patch_size * channels
    keys = jax.random.split(key, 12)

    def normal(k, shape, scale):
        return jax.random.normal(k, shape) * scale

    hidden = d * MLP_RATIO
    return {
        "embed": {
            "w": normal(keys[0], (patch_dim, d), 1.0 / np.sqrt(patch_dim)),
            "b": jnp.zeros((d,)),
        },
        "index_embed": normal(keys[1], (max_tokens, d), 0.02),
        "encoder": {
            "ln1_scale": jnp.ones((d,)),
            "ln1_bias": jnp.zeros((d,)),
            "wq": normal(keys[2], (d, d), 0.02),
            "wk": normal(keys[3], (d, d), 0.02),
            "wv": normal(keys[4], (d, d), 0.02),
            "wo": normal(keys[5], (d, d), 0.02),
            "ln2_scale": jnp.ones((d,)),
            "ln2_bias": jnp.zeros((d,)),
            "mlp_w1": normal(keys[6], (d, hidden), 0.02),
            "mlp_b1": jnp.zeros((hidden,)),
            "mlp_w2": normal(keys[7], (hidden, d), 0.02),
            "mlp_b2": jnp.zeros((d,)),
        },
        "mask_token": normal(keys[8], (d,), 0.02),
        "decoder": {
            "w": normal(keys[9], (d, patch_dim), 0.02),
            "b": jnp.zeros((patch_dim,)),
        },
    }


def _layernorm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def _attention(params, x):
    batch, length, width = x.shape
    head_dim = width // N_HEADS
    query = jnp.matmul(x, params["wq"])
    key = jnp.matmul(x, params["wk"])
    value = jnp.matmul(x, params["wv"])

    def split_heads(tensor):
        tensor = jnp.reshape(tensor, (batch, length, N_HEADS, head_dim))
        return jnp.transpose(tensor, (0, 2, 1, 3))

    query, key, value = split_heads(query), split_heads(key), split_heads(value)
    logits = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(head_dim)
    weights = jax.nn.softmax(logits, axis=-1)
    mixed = jnp.matmul(weights, value)
    mixed = jnp.transpose(mixed, (0, 2, 1, 3))
    mixed = jnp.reshape(mixed, (batch, length, width))
    return jnp.matmul(mixed, params["wo"])


def _encoder_block(params, x):
    hidden = x + _attention(params, _layernorm(x, params["ln1_scale"], params["ln1_bias"]))
    normed = _layernorm(hidden, params["ln2_scale"], params["ln2_bias"])
    mlp = jnp.matmul(jax.nn.gelu(jnp.matmul(normed, params["mlp_w1"]) + params["mlp_b1"]), params["mlp_w2"])
    mlp = mlp + params["mlp_b2"]
    return hidden + mlp


def _forward_loss(params, images, targets, tokens, mask):
    images = _ensure_nhwc(images)
    targets = _ensure_nhwc(targets)
    img_patches = extract_patches(images, tokens)
    tgt_patches = extract_patches(targets, tokens)
    vis_idx = np.flatnonzero(~np.asarray(mask, dtype=bool))
    msk_idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if vis_idx.size == 0 or msk_idx.size == 0:
        raise ValueError("mask must leave at least one visible and one masked token")
    vis_idx_j = jnp.asarray(vis_idx, dtype=jnp.int32)
    msk_idx_j = jnp.asarray(msk_idx, dtype=jnp.int32)

    vis_patches = img_patches[:, vis_idx_j, :]
    embedded = jnp.matmul(vis_patches, params["embed"]["w"]) + params["embed"]["b"]
    embedded = embedded + params["index_embed"][vis_idx_j][None, :, :]
    encoded = _encoder_block(params["encoder"], embedded)
    context = jnp.mean(encoded, axis=1)

    batch, _n_tokens, _patch_dim = img_patches.shape
    width = encoded.shape[-1]
    n_masked = msk_idx_j.shape[0]
    sequence = jnp.zeros((batch, img_patches.shape[1], width), dtype=encoded.dtype)
    sequence = sequence.at[:, vis_idx_j, :].set(encoded)
    masked_repr = params["mask_token"][None, None, :] + context[:, None, :]
    masked_repr = jnp.broadcast_to(masked_repr, (batch, n_masked, width))
    sequence = sequence.at[:, msk_idx_j, :].set(masked_repr)
    sequence = sequence + params["index_embed"][: img_patches.shape[1]][None, :, :]

    pred = jnp.matmul(sequence, params["decoder"]["w"]) + params["decoder"]["b"]
    residual = pred[:, msk_idx_j, :] - tgt_patches[:, msk_idx_j, :]
    return jnp.mean(residual ** 2)


def mae_loss(params, images, salience, key, targets=None, target_n=DEFAULT_TARGET_N, ratio=DEFAULT_RATIO):
    """MSE on masked patches. ``targets`` defaults to ``images``.

    The encoder reads ``images`` only. Changing visible pixels in ``targets``
    does not change the loss; changing masked target pixels does.
    """
    images = _ensure_nhwc(images)
    if targets is None:
        targets = images
    else:
        targets = _ensure_nhwc(targets)
    tokens, _depths, mask, _variances = prepare(salience, key, target_n=target_n, ratio=ratio)
    return _forward_loss(params, images, targets, tokens, mask)


def train_step(params, images, salience, key, lr=1e-3, target_n=DEFAULT_TARGET_N, ratio=DEFAULT_RATIO):
    """One SGD step on the masked reconstruction loss.

    Returns ``(new_params, loss)``. ``loss`` is finite on a non-trivial batch,
    and encoder gradients are taken through visible tokens only.
    """
    images = _ensure_nhwc(images)
    tokens, _depths, mask, _variances = prepare(salience, key, target_n=target_n, ratio=ratio)

    def loss_fn(current):
        return _forward_loss(current, images, images, tokens, mask)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_params = jax.tree_util.tree_map(lambda value, grad: value - lr * grad, params, grads)
    return new_params, loss
