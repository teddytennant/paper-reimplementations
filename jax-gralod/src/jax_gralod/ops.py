"""Shared NHWC primitives: SimpleGate, SCA, convs, anti-aliased resize, box filter."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict[str, Any]

BINOMIAL_1D = (0.25, 0.5, 0.25)


def simple_gate(x: Array) -> Array:
    """NAFNet SimpleGate: split last dim in half and multiply (Chen et al. 2022)."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"SimpleGate needs an even channel count, got {x.shape[-1]}")
    a, b = jnp.split(x, 2, axis=-1)
    return a * b


def init_conv2d(
    key: Array,
    kh: int,
    kw: int,
    in_ch: int,
    out_ch: int,
    groups: int = 1,
    scale: float | None = None,
) -> Params:
    if in_ch % groups != 0:
        raise ValueError(f"in_ch={in_ch} not divisible by groups={groups}")
    in_per = in_ch // groups
    fan_in = kh * kw * in_per
    if scale is None:
        scale = fan_in ** -0.5
    w = scale * jax.random.normal(key, (kh, kw, in_per, out_ch))
    b = jnp.zeros((out_ch,), dtype=w.dtype)
    return {"w": w, "b": b}


def conv2d(params: Params, x: Array, stride: int = 1, groups: int = 1) -> Array:
    y = jax.lax.conv_general_dilated(
        x,
        params["w"],
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=groups,
    )
    return y + params["b"].reshape((1, 1, 1, -1))


def conv1x1(params: Params, x: Array) -> Array:
    return conv2d(params, x, stride=1, groups=1)


def conv3x3(params: Params, x: Array, stride: int = 1) -> Array:
    return conv2d(params, x, stride=stride, groups=1)


def dwconv3x3(params: Params, x: Array) -> Array:
    groups = x.shape[-1]
    return conv2d(params, x, stride=1, groups=groups)


def init_dwconv3x3(key: Array, channels: int, scale: float | None = None) -> Params:
    return init_conv2d(key, 3, 3, channels, channels, groups=channels, scale=scale)


def sca(params: Params, x: Array) -> Array:
    """Simplified channel attention: global spatial mean, 1x1 conv, multiply."""
    gap = jnp.mean(x, axis=(1, 2), keepdims=True)
    return x * conv1x1(params, gap)


def layer_norm(params: Params, x: Array, eps: float = 1e-5) -> Array:
    """Epsilon-safe LayerNorm over the channel axis."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + eps)
    return y * params["scale"] + params["bias"]


def init_layer_norm(channels: int) -> Params:
    return {
        "scale": jnp.ones((channels,), dtype=jnp.float32),
        "bias": jnp.zeros((channels,), dtype=jnp.float32),
    }


def relu(x: Array) -> Array:
    return jnp.maximum(x, 0.0)


def binomial_blur(x: Array) -> Array:
    """Separable [1, 2, 1]/4 blur with edge padding."""
    a, b, c = BINOMIAL_1D
    xp = jnp.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
    xh = a * xp[:, 0:-2] + b * xp[:, 1:-1] + c * xp[:, 2:]
    xp = jnp.pad(xh, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
    return a * xp[:, :, 0:-2] + b * xp[:, :, 1:-1] + c * xp[:, :, 2:]


def antialiased_resize(x: Array, height: int, width: int) -> Array:
    """Resize NHWC features. Downsampling blurs with a binomial kernel first."""
    b, h, w, ch = x.shape
    if h == height and w == width:
        return x
    reducing = h > height or w > width
    if reducing:
        x = binomial_blur(x)
        scale_h = h / max(height, 1)
        scale_w = w / max(width, 1)
        if scale_h >= 3.5 or scale_w >= 3.5:
            x = binomial_blur(x)
        if h == 2 * height and w == 2 * width:
            return x[:, ::2, ::2, :]
    return jax.image.resize(x, (b, height, width, ch), method="bilinear")


def box_filter(x: Array, ksize: int = 5) -> Array:
    """Same-size local average (A_r). Default 5x5 as in Appendix A.2.2."""
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"box_filter ksize must be a positive odd int, got {ksize}")
    pad = ksize // 2
    xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    summed = jax.lax.reduce_window(
        xp,
        0.0,
        jax.lax.add,
        window_dimensions=(1, ksize, ksize, 1),
        window_strides=(1, 1, 1, 1),
        padding="VALID",
    )
    return summed / float(ksize * ksize)


def four_connected_offsets() -> tuple[tuple[int, int], ...]:
    """4-connected neighbor offsets: left, right, up, down."""
    return ((0, -1), (0, 1), (-1, 0), (1, 0))
