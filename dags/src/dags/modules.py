"""Thin stand-in encoders, per-layer scales, and one stabilizer step.

Stem widths, the stabilizer conv, and the 3x3 box blur are not given in the
paper. They are stand-ins. There is no U-Net here, and facetization is omitted.
"""

import jax
import jax.numpy as jnp

from dags.control import inject, lighting_recurrence


def conv2d(x, kernel, bias=None):
    """Same-padded conv. x is (..., H, W, C_in), kernel is (K, K, C_in, C_out)."""
    x = jnp.asarray(x)
    kernel = jnp.asarray(kernel)
    lead = x.shape[:-3]
    height, width, channels = x.shape[-3:]
    flat = jnp.reshape(x, (-1, height, width, channels))
    y = jax.lax.conv_general_dilated(
        flat,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    if bias is not None:
        y = y + jnp.asarray(bias)
    return jnp.reshape(y, lead + y.shape[-3:])


def box_blur_3x3(image):
    """Fixed 3x3 box blur. The paper's kernel size is unpublished; 3x3 is a stand-in.

    Each channel is averaged with its 3x3 neighborhood. Samples outside the
    frame are zero, so the kernel sums to 1/9 everywhere (borders are darker).
    """
    image = jnp.asarray(image)
    lead = image.shape[:-3]
    height, width, channels = image.shape[-3:]
    flat = jnp.reshape(image, (-1, height, width, channels))
    padded = jnp.pad(flat, ((0, 0), (1, 1), (1, 1), (0, 0)))
    acc = jnp.zeros_like(flat)
    for dy in range(3):
        for dx in range(3):
            acc = acc + padded[:, dy : dy + height, dx : dx + width, :]
    acc = acc / 9.0
    return jnp.reshape(acc, image.shape)


def _normal(key, shape, std):
    return jax.random.normal(key, shape) * std


class GeometryEncoder:
    """Two conv stems, depth (1 channel) and normals (3 channels).

    Each stem is a 3x3 conv to 8 channels with SiLU. The concat (16 channels)
    goes through a 1x1 conv to d_out. Output is (..., H, W, d_out).
    The stems do not share parameters.
    """

    stem_width = 8

    @staticmethod
    def init(key, d_out, std=0.02):
        k_depth, k_normal, k_proj = jax.random.split(key, 3)
        width = GeometryEncoder.stem_width
        return {
            "depth_stem": {
                "kernel": _normal(k_depth, (3, 3, 1, width), std),
                "bias": jnp.zeros((width,)),
            },
            "normal_stem": {
                "kernel": _normal(k_normal, (3, 3, 3, width), std),
                "bias": jnp.zeros((width,)),
            },
            "proj": {
                "kernel": _normal(k_proj, (1, 1, 2 * width, int(d_out)), std),
                "bias": jnp.zeros((int(d_out),)),
            },
        }

    @staticmethod
    def apply(params, depth, normals):
        depth = jnp.asarray(depth)
        normals = jnp.asarray(normals)
        if depth.shape[-1] != 1:
            raise ValueError("depth must be (..., H, W, 1)")
        if normals.shape[-1] != 3:
            raise ValueError("normals must be (..., H, W, 3)")
        d = jax.nn.silu(conv2d(depth, params["depth_stem"]["kernel"], params["depth_stem"]["bias"]))
        n = jax.nn.silu(conv2d(normals, params["normal_stem"]["kernel"], params["normal_stem"]["bias"]))
        fused = jnp.concatenate([d, n], axis=-1)
        return conv2d(fused, params["proj"]["kernel"], params["proj"]["bias"])


class AppearanceEncoder:
    """One 3x3 conv on concat(albedo, raw 1-spp, R).

    Three RGB inputs make 9 channels. Output is (..., H, W, d_out).
    """

    @staticmethod
    def init(key, d_out, in_channels=9, std=0.02):
        return {
            "kernel": _normal(key, (3, 3, int(in_channels), int(d_out)), std),
            "bias": jnp.zeros((int(d_out),)),
        }

    @staticmethod
    def apply(params, albedo, raw, relit):
        x = jnp.concatenate([jnp.asarray(albedo), jnp.asarray(raw), jnp.asarray(relit)], axis=-1)
        return conv2d(x, params["kernel"], params["bias"])


def token_align(feat, n_tokens_side):
    """Average-pool a feature map to an n_side by n_side grid and flatten.

    feat is (..., H, W, D). Output is (..., n_side * n_side, D), row-major.
    This is a stand-in for the paper's token-aligned injection, not its patch embed.
    Bins use the usual adaptive split when H or W is not divisible by n_side.
    """
    feat = jnp.asarray(feat)
    n = int(n_tokens_side)
    if n < 1:
        raise ValueError("n_tokens_side must be positive")
    *lead, height, width, channels = feat.shape
    if height < 1 or width < 1:
        raise ValueError("feature map must be non-empty")

    if height % n == 0 and width % n == 0:
        hs = height // n
        ws = width // n
        blocks = jnp.reshape(feat, lead + [n, hs, n, ws, channels])
        pooled = jnp.mean(blocks, axis=(-4, -2))
    else:
        ys = jnp.arange(height)
        xs = jnp.arange(width)
        y_bin = (ys * n) // height
        x_bin = (xs * n) // width
        y_oh = (y_bin[:, None] == jnp.arange(n)[None, :]).astype(feat.dtype)
        x_oh = (x_bin[:, None] == jnp.arange(n)[None, :]).astype(feat.dtype)
        y_count = jnp.sum(y_oh, axis=0)
        x_count = jnp.sum(x_oh, axis=0)
        count = jnp.maximum(y_count[:, None] * x_count[None, :], 1.0)
        tmp = jnp.einsum("...hwd,hi->...iwd", feat, y_oh)
        summed = jnp.einsum("...iwd,wj->...ijd", tmp, x_oh)
        pooled = summed / count[..., None]

    return jnp.reshape(pooled, lead + [n * n, channels])


def init_layer_scales(n_layers):
    """Per-layer injection scales, initialized at 0 so injection starts as a no-op.

    Returns a pytree with s_g[l] and s_a[l] for l in 0 .. n_layers-1.
    """
    n = int(n_layers)
    if n < 1:
        raise ValueError("n_layers must be positive")
    return {
        "s_g": jnp.zeros((n,)),
        "s_a": jnp.zeros((n,)),
    }


def inject_layers(h, c_g, c_a, scales, image_mask):
    """Apply a different (s_g, s_a) pair at each layer.

    h is one array, reused at every layer, or a list of n_layers arrays.
    Always returns a list of n_layers arrays. c_g and c_a are shared across
    layers (once-per-frame encodings). Changing s_g[i] does not affect layer j.
    """
    s_g = jnp.asarray(scales["s_g"])
    s_a = jnp.asarray(scales["s_a"])
    n = int(s_g.shape[0])
    if int(s_a.shape[0]) != n:
        raise ValueError("s_g and s_a must have the same length")
    if isinstance(h, (list, tuple)):
        if len(h) != n:
            raise ValueError(f"expected {n} layer states, got {len(h)}")
        states = list(h)
    else:
        states = [h] * n
    return [inject(states[i], c_g, c_a, s_g[i], s_a[i], image_mask) for i in range(n)]


class StabilizerStep:
    """One recurrent lighting step with a tiny conv stand-in for the U-Net.

    A fixed 3x3 box blur is applied to the current demodulated frame, then a
    3x3 conv maps C channels to 3C. The conv output is split into L_star, a
    pre-sigmoid gate, and delta. The gate is passed through a sigmoid and the
    exact lighting recurrence is applied. Kernel init scale is std 0.02.
    """

    @staticmethod
    def init(key, channels, std=0.02):
        channels = int(channels)
        kernel = _normal(key, (3, 3, channels, 3 * channels), std)
        bias = jnp.zeros((3 * channels,))
        return {"kernel": kernel, "bias": bias}

    @staticmethod
    def apply(params, frame, motion, valid, L_prev):
        frame = jnp.asarray(frame)
        blurred = box_blur_3x3(frame)
        pred = conv2d(blurred, params["kernel"], params["bias"])
        channels = frame.shape[-1]
        L_star = pred[..., :channels]
        gate = jax.nn.sigmoid(pred[..., channels : 2 * channels])
        delta = pred[..., 2 * channels :]
        return lighting_recurrence(L_star, gate, L_prev, delta, motion, valid)
