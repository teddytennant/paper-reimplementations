"""Symmetric per-channel quantization and sequential GLF reconstruction."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def quantize_symmetric(w, bits=4, axis=-1):
    """Per-channel symmetric uniform quantizer. Round is identity in backward."""
    qmax = 2 ** (int(bits) - 1) - 1
    w = jnp.asarray(w, dtype=jnp.float32)
    amax = jnp.max(jnp.abs(w), axis=axis, keepdims=True)
    scale = amax / qmax
    scale_safe = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    scaled = w / scale_safe
    q = jnp.clip(jnp.round(scaled), -qmax, qmax)
    q_ste = scaled + jax.lax.stop_gradient(q - scaled)
    return q_ste * scale_safe


def glf_loss(local_mse, global_mse, local0, global0):
    """L = L_g / sg(L_g0) + L_l / sg(L_l0). A zero init term uses 1."""
    local0 = jnp.asarray(local0, dtype=jnp.float32)
    global0 = jnp.asarray(global0, dtype=jnp.float32)
    local_den = jnp.where(local0 == 0, jnp.ones_like(local0), local0)
    global_den = jnp.where(global0 == 0, jnp.ones_like(global0), global0)
    return global_mse / jax.lax.stop_gradient(global_den) + local_mse / jax.lax.stop_gradient(
        local_den
    )


def predict(blocks, x):
    """blocks is a list of (W, b). y = x @ W + b, in order."""
    h = jnp.asarray(x, dtype=jnp.float32)
    for weight, bias in blocks:
        h = h @ jnp.asarray(weight, dtype=jnp.float32) + jnp.asarray(bias, dtype=jnp.float32)
    return h


def _split(blocks):
    weights = [jnp.asarray(weight, dtype=jnp.float32) for weight, _bias in blocks]
    biases = [jnp.asarray(bias, dtype=jnp.float32) for _weight, bias in blocks]
    return weights, biases


def _qmax(bits):
    return float(2 ** (int(bits) - 1) - 1)


def _quantize_ste(weight, scale, bits):
    """Forward is round(w/s)*s. Scale grad flows through the code; weight grad is identity."""
    qmax = _qmax(bits)
    codes = jnp.clip(jnp.round(weight / scale), -qmax, qmax)
    hat = jax.lax.stop_gradient(codes) * scale
    return hat + (weight - jax.lax.stop_gradient(weight))


def _frozen(weight, scale, bits):
    qmax = _qmax(bits)
    codes = jnp.clip(jnp.round(weight / scale), -qmax, qmax)
    return jax.lax.stop_gradient(codes * scale)


def _init_scale(weight, bits):
    qmax = _qmax(bits)
    scale = jnp.max(jnp.abs(weight), axis=-1, keepdims=True) / qmax
    return jnp.maximum(scale, 1e-8)


def _forward_mixed(x0, weights, biases, prefix_q, index, used):
    hidden = x0
    local = None
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if i < index:
            current = prefix_q[i]
        elif i == index:
            current = used
        else:
            current = weight
        hidden = hidden @ current + bias
        if i == index:
            local = hidden
    return local, hidden


def block_glf_loss(
    blocks,
    x0,
    prefix_q,
    index,
    weight,
    bits=4,
    use_global=True,
    lam=2.0,
    quantize=True,
):
    """Normalized GLF loss of one block against round-to-nearest initialization.

    prefix_q holds frozen quantized weights before index. quantize=False scores
    an already reconstructed matrix instead of rounding it again.
    """
    weights, biases = _split(blocks)
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    hidden = x0
    fp_outs = []
    for fp_weight, bias in zip(weights, biases):
        hidden = hidden @ fp_weight + bias
        fp_outs.append(hidden)

    def terms(candidate, do_quant):
        if do_quant:
            used = quantize_symmetric(candidate, bits=bits, axis=-1)
        else:
            used = candidate
        local, final = _forward_mixed(x0, weights, biases, prefix_q, index, used)
        local_mse = jnp.mean((local - fp_outs[index]) ** 2)
        global_mse = jnp.mean((final - fp_outs[-1]) ** 2)
        if not use_global:
            global_mse = jnp.zeros_like(global_mse)
        return local_mse, global_mse

    local0, global0 = terms(weights[index], True)
    local0 = jnp.maximum(local0, 1e-12)
    global0 = jnp.maximum(global0, 1e-12)
    if not use_global:
        global0 = jnp.ones_like(global0)
    local_mse, global_mse = terms(weight, quantize)
    return glf_loss(lam * local_mse, global_mse, local0, global0)


def reconstruct(blocks, x0, bits=4, steps=100, lr=2e-5, seed=0, use_global=True, scale_lr=2e-4, lam=2.0):
    """Optimize each block in order, then freeze its quantized weights.

    Adam updates the block weights and the per-channel scale through STE.
    The objective is the paper's normalized global feature loss plus lambda
    times the local feature loss. Returns one quantized weight matrix per block.
    Deterministic given seed.
    """
    weights, biases = _split(blocks)
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    _ = jax.random.PRNGKey(int(seed))

    hidden = x0
    fp_outs = []
    for weight, bias in zip(weights, biases):
        hidden = hidden @ weight + bias
        fp_outs.append(hidden)

    quantized = []
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for index, weight0 in enumerate(weights):
        prefix = list(quantized)
        scale = _init_scale(weight0, bits)

        def loss_of(weight_var, scale_var, index=index, prefix=prefix):
            used = _quantize_ste(weight_var, scale_var, bits)
            local, final = _forward_mixed(x0, weights, biases, prefix, index, used)
            local_mse = jnp.mean((local - fp_outs[index]) ** 2)
            global_mse = jnp.mean((final - fp_outs[-1]) ** 2)
            if not use_global:
                global_mse = jnp.zeros_like(global_mse)
            return local_mse, global_mse

        local0, global0 = loss_of(weight0, scale)
        local0 = jax.lax.stop_gradient(jnp.maximum(local0, 1e-12))
        global0 = jax.lax.stop_gradient(jnp.maximum(global0, 1e-12))
        if not use_global:
            global0 = jnp.ones_like(global0)

        def objective(weight_var, scale_var, local0=local0, global0=global0):
            local_mse, global_mse = loss_of(weight_var, scale_var)
            return glf_loss(lam * local_mse, global_mse, local0, global0)

        moment_w = jnp.zeros_like(weight0)
        second_w = jnp.zeros_like(weight0)
        moment_s = jnp.zeros_like(scale)
        second_s = jnp.zeros_like(scale)
        weight = weight0
        for step in range(1, int(steps) + 1):
            _loss, grads = jax.value_and_grad(objective, argnums=(0, 1))(weight, scale)
            grad_w, grad_s = grads
            moment_w = beta1 * moment_w + (1.0 - beta1) * grad_w
            second_w = beta2 * second_w + (1.0 - beta2) * (grad_w * grad_w)
            moment_s = beta1 * moment_s + (1.0 - beta1) * grad_s
            second_s = beta2 * second_s + (1.0 - beta2) * (grad_s * grad_s)
            weight = weight - lr * (moment_w / (1.0 - beta1 ** step)) / (
                jnp.sqrt(second_w / (1.0 - beta2 ** step)) + eps
            )
            scale = scale - scale_lr * (moment_s / (1.0 - beta1 ** step)) / (
                jnp.sqrt(second_s / (1.0 - beta2 ** step)) + eps
            )
            scale = jnp.maximum(scale, 1e-8)

        quantized.append(_frozen(weight, scale, bits))
    return quantized
