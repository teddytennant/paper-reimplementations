"""Control equations for DAGS (arXiv 2610.02567).

Pure functions. No learned parameters.

Demodulation, remodulation, backward warp, the lighting recurrence, temporal
guidance, the flow source, and token injection follow the written equations.
Specular-motion modulation and Delaunay facetization are not given as formulas
and are not implemented.
"""

import jax
import jax.numpy as jnp


def demodulate(image, albedo, eps=1e-4):
    """Log-ratio lighting: L = log(clip(I / a, min=eps)).

    image and albedo are positive and shaped (..., H, W, C). Elementwise.
    """
    image = jnp.asarray(image)
    albedo = jnp.asarray(albedo)
    ratio = image / albedo
    return jnp.log(jnp.clip(ratio, min=eps))


def remodulate(albedo, lighting):
    """Flat color from a lighting map: I_hat = a * exp(L). Elementwise."""
    return jnp.asarray(albedo) * jnp.exp(jnp.asarray(lighting))


def _bilinear_sample(image, sx, sy):
    """Sample image (H, W, C) at floating pixel coords. Zero outside the frame."""
    height, width, _channels = image.shape
    x0 = jnp.floor(sx).astype(jnp.int32)
    y0 = jnp.floor(sy).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (sx - x0.astype(sx.dtype))[..., None]
    wy = (sy - y0.astype(sy.dtype))[..., None]

    def gather(yy, xx):
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy_c = jnp.clip(yy, 0, height - 1)
        xx_c = jnp.clip(xx, 0, width - 1)
        vals = image[yy_c, xx_c]
        return jnp.where(valid[..., None], vals, 0)

    v00 = gather(y0, x0)
    v01 = gather(y0, x1)
    v10 = gather(y1, x0)
    v11 = gather(y1, x1)
    return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)


def _warp_one(image, motion):
    """Backward warp one frame. motion is (H, W, 2) as (dx, dy) in pixels."""
    height, width, _channels = image.shape
    dtype = image.dtype
    ys = jnp.arange(height, dtype=dtype)
    xs = jnp.arange(width, dtype=dtype)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    # Sample the source at (x - dx, y - dy).
    sy = grid_y - motion[..., 1]
    sx = grid_x - motion[..., 0]
    return _bilinear_sample(image, sx, sy)


def backward_warp(image, motion):
    """Backward warp.

    image is (..., H, W, C). motion is (..., H, W, 2) in pixels, (dx, dy).
    The output at (x, y) is image sampled at (x - dx, y - dy), bilinear, and
    zero when the sample falls outside the frame.
    """
    image = jnp.asarray(image)
    motion = jnp.asarray(motion)
    if image.ndim < 3 or motion.ndim < 3 or motion.shape[-1] != 2:
        raise ValueError("image is (..., H, W, C) and motion is (..., H, W, 2)")
    height, width, channels = image.shape[-3], image.shape[-2], image.shape[-1]
    if motion.shape[-3] != height or motion.shape[-2] != width:
        raise ValueError("motion spatial size must match image")
    batch = jnp.broadcast_shapes(image.shape[:-3], motion.shape[:-3])
    image_b = jnp.broadcast_to(image, batch + (height, width, channels))
    motion_b = jnp.broadcast_to(motion, batch + (height, width, 2))
    flat_i = jnp.reshape(image_b, (-1, height, width, channels))
    flat_m = jnp.reshape(motion_b, (-1, height, width, 2))
    # vmap over a size-1 batch still works when there is no leading dim.
    warped = jax.vmap(_warp_one)(flat_i, flat_m)
    return jnp.reshape(warped, image_b.shape)


def _as_channel_mask(mask, ref):
    """Broadcast a spatial or token mask onto ref, adding a trailing axis if needed."""
    mask = jnp.asarray(mask)
    if mask.dtype != ref.dtype:
        mask = mask.astype(ref.dtype)
    if mask.ndim == ref.ndim - 1:
        mask = mask[..., None]
    return mask


def lighting_recurrence(L_star, g, L_prev, delta, motion, valid):
    """Recurrent lighting fuse.

    Where valid:
        L_t = L_star + g * (warp(L_prev) - L_star) + delta
    Where not valid (disoccluded pixels):
        L_t = L_star

    g is the learned gate. Nonzero entries of valid are treated as valid.
    valid is (..., H, W) or (..., H, W, 1) and is broadcast over channels.
    Invalid pixels are exactly L_star: no gate term and no delta.
    """
    L_star = jnp.asarray(L_star)
    g = jnp.asarray(g)
    delta = jnp.asarray(delta)
    warped = backward_warp(L_prev, motion)
    fused = L_star + g * (warped - L_star) + delta
    mask = jnp.asarray(valid)
    if mask.ndim == L_star.ndim - 1:
        mask = mask[..., None]
    mask = jnp.broadcast_to(mask, fused.shape)
    return jnp.where(mask != 0, fused, L_star)


def temporal_lambda(sigma, lam_max, sigma_max):
    """lam(sigma) = lam_max * relu(1 - sigma / sigma_max)."""
    sigma = jnp.asarray(sigma)
    lam_max = jnp.asarray(lam_max)
    sigma_max = jnp.asarray(sigma_max)
    return lam_max * jnp.maximum(1.0 - sigma / sigma_max, 0.0)


def temporal_guidance(x_hat, x_prev, motion, mask, sigma, lam_max, sigma_max):
    """Temporally guided denoising step.

    x_hat = x_hat + lam(sigma) * M * (warp(x_prev) - x_hat)

    M is an occlusion or validity mask and is broadcast over channels.
    """
    x_hat = jnp.asarray(x_hat)
    lam = temporal_lambda(sigma, lam_max, sigma_max)
    warped = backward_warp(x_prev, motion)
    mask = _as_channel_mask(mask, x_hat)
    return x_hat + lam * mask * (warped - x_hat)


def flow_source(R):
    """Diffusion flow source x_0 is the coarse relit prior R, not Gaussian noise."""
    return R


def inject(h, c_g, c_a, s_g, s_a, image_mask):
    """Conditioning injection, elementwise over tokens.

    h_out = h + image_mask * (s_g * c_g + s_a * c_a)

    h, c_g, c_a are (..., N, D). s_g and s_a are scalars. image_mask is
    (..., N) or (..., N, 1): 1 on image tokens, 0 on text tokens. Text tokens
    are unchanged. This is not attention: token i depends only on token i.
    """
    h = jnp.asarray(h)
    c_g = jnp.asarray(c_g)
    c_a = jnp.asarray(c_a)
    mask = _as_channel_mask(image_mask, h)
    residual = s_g * c_g + s_a * c_a
    return h + mask * residual
