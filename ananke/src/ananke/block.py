"""CTAN-Res block and a tiny classifier smoke model.

Equation 22:

    x_sp = S(x)
    u = W_in LayerNorm(x_sp)
    x_next = x + DropPath(gamma * W_out Psi_{t, eta}(u))

S is a sum of 3x3 and 5x5 depthwise convolutions. That is a stand-in for the
paper's multi-scale diffusion operator, whose kernels are not specified.
W_in and W_out are pointwise maps. Psi is the exact log-symplectic flow on
consecutive channel pairs of u. gamma is a per-channel scale, initialized small.
t is a positive per-layer scalar (softplus of a learned raw value).

The context generator is a 1x1 convolution. The paper does not specify the
layers inside G. See module docstring of context_parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ananke.flow import exact_circular_flow


_CONV_DIMS = ("NCHW", "OIHW", "NCHW")


def _softplus_inverse(y):
    y = jnp.asarray(y)
    return jnp.log(jnp.expm1(y))


def layer_norm(x, scale, bias, eps=1e-5):
    """Channel LayerNorm at each spatial site. x is NCHW."""
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=1, keepdims=True)
    y = (x - mean) * jnp.reciprocal(jnp.sqrt(var + eps))
    scale = scale.reshape(1, -1, 1, 1)
    bias = bias.reshape(1, -1, 1, 1)
    return y * scale + bias


def pointwise(x, weight, bias):
    """1x1 projection. weight is (out_channels, in_channels)."""
    y = jnp.einsum("oi,nihw->nohw", weight, x)
    return y + bias.reshape(1, -1, 1, 1)


def depthwise_conv(x, weight):
    """Depthwise conv, NCHW. weight is (channels, 1, k, k)."""
    channels = x.shape[1]
    return jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=_CONV_DIMS,
        feature_group_count=channels,
    )


def conv2d(x, weight, bias):
    """Standard conv, NCHW, stride 1, same padding. weight is OIHW."""
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=_CONV_DIMS,
    )
    return y + bias.reshape(1, -1, 1, 1)


def drop_path(x, drop_prob, rng, deterministic):
    """Stochastic depth. Per-sample mask, inverted dropout scaling.

    drop_prob == 0 or deterministic=True returns x unchanged.
    """
    if deterministic or drop_prob == 0.0:
        return x
    if rng is None:
        raise ValueError("rng is required when DropPath is stochastic")
    keep = 1.0 - drop_prob
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jax.random.bernoulli(rng, keep, shape=mask_shape).astype(x.dtype)
    return x * mask / jnp.asarray(keep, dtype=x.dtype)


def diffusion(params, x):
    """Multi-scale depthwise stand-in: S(x) = DW_3(x) + DW_5(x)."""
    return depthwise_conv(x, params["dw3"]) + depthwise_conv(x, params["dw5"])


def context_parameters(params, u):
    """Predict frozen flow parameters from u, once per site and plane.

    The paper writes eta = G(X) = (c, R, beta, alpha_eff, a) and does not
    specify the layers inside G. Here G is a 1x1 conv from D channels to 6K
    channels, split per plane into (c_x, c_y, raw_R, raw_beta, p_alpha, a).

    R = softplus(raw_R) + floor, beta = softplus(raw_beta) + floor.
    alpha_eff = alpha0 * (1 + 0.5 * tanh(p_alpha)), with alpha0 a learned
    per-plane base frequency (not spatial). Parameters are computed once and
    held fixed for the closed-form step, which is the freeze the paper asks for.

    G is applied to u, the tensor Psi consumes, so c lives in the same
    coordinates as the flowed features. The paper does not say whether X is
    the block input or the projected features.
    """
    raw = pointwise(u, params["g_w"], params["g_b"])
    k = params["alpha0"].shape[0]
    cx, cy, raw_r, raw_beta, p_alpha, twist = jnp.split(raw, 6, axis=1)
    if cx.shape[1] != k:
        raise ValueError("context generator width does not match K phase planes")
    floor = jnp.asarray(1e-3, dtype=u.dtype)
    radius = jax.nn.softplus(raw_r) + floor
    rate = jax.nn.softplus(raw_beta) + floor
    alpha0 = params["alpha0"].reshape(1, k, 1, 1)
    alpha_eff = alpha0 * (1.0 + 0.5 * jnp.tanh(p_alpha))
    center = jnp.stack([cx, cy], axis=2)  # N, K, 2, H, W
    return center, radius, rate, alpha_eff, twist


def psi_elsf(u, center, radius, rate, alpha_eff, twist, t, radius_eps):
    """Exact flow on consecutive pairs. u is NCHW with even channel count."""
    n, d, h, w = u.shape
    k = d // 2
    planes = u.reshape(n, k, 2, h, w)
    planes = jnp.transpose(planes, (0, 3, 4, 1, 2))  # N, H, W, K, 2
    center = jnp.transpose(center, (0, 3, 4, 1, 2))
    radius = jnp.transpose(radius, (0, 2, 3, 1))
    rate = jnp.transpose(rate, (0, 2, 3, 1))
    alpha_eff = jnp.transpose(alpha_eff, (0, 2, 3, 1))
    twist = jnp.transpose(twist, (0, 2, 3, 1))
    y = exact_circular_flow(
        planes,
        center,
        radius,
        rate,
        alpha_eff,
        twist,
        t,
        eps=radius_eps,
        rotation="exact",
    )
    y = jnp.transpose(y, (0, 3, 4, 1, 2))  # N, K, 2, H, W
    return y.reshape(n, d, h, w)


def init_ctan_res(key, channels, dtype=jnp.float32):
    """Initialize one CTAN-Res block. channels = D = 2K must be even.

    Depthwise 3x3 starts as identity, 5x5 starts at 0, so S starts as the
    identity. W_in and W_out start as identity. gamma starts at 1e-2.
    Context-generator weights are small; R and beta biases target 1.
    """
    if channels % 2 != 0:
        raise ValueError("channels must be even (D = 2K)")
    k = channels // 2
    dtype = jnp.dtype(dtype)
    key_g, key_a = jax.random.split(key)
    dw3 = jnp.zeros((channels, 1, 3, 3), dtype=dtype)
    dw3 = dw3.at[:, 0, 1, 1].set(1.0)
    dw5 = jnp.zeros((channels, 1, 5, 5), dtype=dtype)
    eye = jnp.eye(channels, dtype=dtype)
    g_w = jax.random.normal(key_g, (6 * k, channels), dtype=dtype) * 0.02
    g_b = jnp.zeros((6 * k,), dtype=dtype)
    # Layout: c_x, c_y, raw_R, raw_beta, p_alpha, a. Each block has length K.
    r_bias = _softplus_inverse(jnp.asarray(1.0, dtype=dtype))
    g_b = g_b.at[2 * k : 3 * k].set(r_bias)
    g_b = g_b.at[3 * k : 4 * k].set(r_bias)
    alpha0 = jnp.ones((k,), dtype=dtype)
    # Tiny jitter so planes are not identical at step 0. Mean stays 1.
    alpha0 = alpha0 + 0.01 * jax.random.normal(key_a, (k,), dtype=dtype)
    return {
        "dw3": dw3,
        "dw5": dw5,
        "ln_scale": jnp.ones((channels,), dtype=dtype),
        "ln_bias": jnp.zeros((channels,), dtype=dtype),
        "w_in": eye,
        "b_in": jnp.zeros((channels,), dtype=dtype),
        "w_out": eye.copy(),
        "b_out": jnp.zeros((channels,), dtype=dtype),
        "gamma": jnp.full((channels,), 1e-2, dtype=dtype),
        "raw_t": jnp.asarray(0.0, dtype=dtype),
        "alpha0": alpha0,
        "g_w": g_w,
        "g_b": g_b,
    }


def ctan_res(params, x, drop_prob=0.0, deterministic=True, rng=None, radius_eps=1e-6):
    """CTAN-Res forward. x is NCHW, output shape equals input shape.

    With gamma = 0 the residual branch is zero, so the block is the identity.
    radius_eps floors ||u - c|| inside Psi so a feature that lands on its
    center does not make log undefined. The pure flow tests leave eps at 0.
    """
    x_sp = diffusion(params, x)
    normalized = layer_norm(x_sp, params["ln_scale"], params["ln_bias"])
    u = pointwise(normalized, params["w_in"], params["b_in"])
    center, radius, rate, alpha_eff, twist = context_parameters(params, u)
    t = jax.nn.softplus(params["raw_t"])
    flowed = psi_elsf(u, center, radius, rate, alpha_eff, twist, t, radius_eps)
    mixed = pointwise(flowed, params["w_out"], params["b_out"])
    gamma = params["gamma"].reshape(1, -1, 1, 1)
    residual = drop_path(mixed * gamma, drop_prob, rng, deterministic)
    return x + residual


def init_tiny_classifier(key, in_channels=3, width=8, n_classes=4, dtype=jnp.float32):
    """Stem conv, two CTAN-Res blocks, global pool, linear head.

    This is a smoke model, not the paper's hierarchical backbone. Width is
    even so channels pair into phase planes. Stem stride is 1 so an 8x8
    image keeps its grid; the paper's stem is stride 2 on larger inputs.
    """
    if width % 2 != 0:
        raise ValueError("width must be even")
    k_stem, k_b1, k_b2, k_head = jax.random.split(key, 4)
    dtype = jnp.dtype(dtype)
    fan_in = in_channels * 9
    stem_w = jax.random.normal(k_stem, (width, in_channels, 3, 3), dtype=dtype)
    stem_w = stem_w * jnp.sqrt(jnp.asarray(2.0 / fan_in, dtype=dtype))
    block1 = init_ctan_res(k_b1, width, dtype=dtype)
    block2 = init_ctan_res(k_b2, width, dtype=dtype)
    head_w = jax.random.normal(k_head, (n_classes, width), dtype=dtype) * 0.02
    return {
        "stem_w": stem_w,
        "stem_b": jnp.zeros((width,), dtype=dtype),
        "block1": block1,
        "block2": block2,
        "head_w": head_w,
        "head_b": jnp.zeros((n_classes,), dtype=dtype),
    }


def tiny_classifier(params, x):
    """Logits for an NCHW batch. DropPath is off (deterministic smoke path)."""
    h = conv2d(x, params["stem_w"], params["stem_b"])
    h = ctan_res(params["block1"], h, drop_prob=0.0, deterministic=True)
    h = ctan_res(params["block2"], h, drop_prob=0.0, deterministic=True)
    pooled = jnp.mean(h, axis=(2, 3))
    return pooled @ params["head_w"].T + params["head_b"]


def softmax_cross_entropy(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    n = labels.shape[0]
    picked = log_probs[jnp.arange(n), labels]
    return -jnp.mean(picked)


def make_synthetic_batch(n=32, image_size=8, in_channels=3, n_classes=4, seed=0, dtype=jnp.float32):
    """Fixed class-conditional 8x8 batch. Class cue is a channel offset plus a corner."""
    key = jax.random.PRNGKey(seed)
    images = jax.random.normal(key, (n, in_channels, image_size, image_size), dtype=dtype)
    images = images * jnp.asarray(0.3, dtype=dtype)
    labels = jnp.arange(n) % n_classes
    cue = labels.astype(dtype)[:, None, None, None]
    images = images.at[:, 0:1].add(0.8 * cue)
    images = images.at[:, 1:2, 0:2, 0:2].add(0.5 * cue)
    return images, labels


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "step": jnp.array(0)}


def _adam_apply(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    step = state["step"] + 1
    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda mi: mi / (1.0 - b1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda vi: vi / (1.0 - b2 ** step), v)
    updated = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return updated, {"m": m, "v": v, "step": step}


def train_tiny_classifier(steps=20, seed=0, lr=0.05, batch=32, width=8, n_classes=4, image_size=8):
    """Adam on one fixed synthetic batch. Returns (loss_at_step_0, loss_at_step_steps).

    CPU smoke only. Not a CIFAR or Kvasir run, and not a reported accuracy.
    """
    key = jax.random.PRNGKey(seed)
    params = init_tiny_classifier(key, width=width, n_classes=n_classes)
    images, labels = make_synthetic_batch(
        n=batch,
        image_size=image_size,
        n_classes=n_classes,
        seed=seed + 1,
        dtype=params["stem_b"].dtype,
    )

    def loss_of(p):
        return softmax_cross_entropy(tiny_classifier(p, images), labels)

    @jax.jit
    def update(p, state):
        loss, grads = jax.value_and_grad(loss_of)(p)
        p, state = _adam_apply(p, grads, state, lr)
        return p, state, loss

    state = _adam_init(params)
    loss0 = loss_of(params)
    for _ in range(steps):
        params, state, _ = update(params, state)
    loss_final = loss_of(params)
    return loss0, loss_final
