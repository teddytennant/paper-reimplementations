"""Timestep-aware gated attention (content gate plus a shared SiLU time bias)."""

import os

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp


def gate_width(d_model, n_heads, granularity):
    """Return the time-bias width D_g for element-wise or head-wise gates."""
    if n_heads <= 0 or d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    d_h = d_model // n_heads
    if d_h % 2 != 0:
        raise ValueError("head dimension must be even")
    if granularity == "element":
        return int(d_model)
    if granularity == "head":
        return int(n_heads)
    raise ValueError("granularity must be 'element' or 'head'")


def time_branch_param_count(n_streams, n_layers, d_model, d_temb, n_heads, granularity):
    """Parameter count of the time branch only: streams * layers * (d_temb * D_g + D_g)."""
    d_g = gate_width(d_model, n_heads, granularity)
    return int(n_streams * n_layers * (d_temb * d_g + d_g))


def _sigmoid(x):
    """Logistic sigmoid, kept strictly inside (0, 1) in low precision."""
    y = jax.nn.sigmoid(x)
    info = jnp.finfo(y.dtype)
    one = jnp.asarray(1, dtype=y.dtype)
    y = jnp.where(y >= one, one - info.eps, y)
    y = jnp.where(y <= 0, info.tiny, y)
    return y


def silu(x):
    """SiLU: u * sigmoid(u)."""
    return x * _sigmoid(x)


def time_bias(temb, w_time, b_time):
    """Token-shared bias B = silu(temb) @ W_time + b_time."""
    return silu(temb) @ w_time + b_time


def _granularity_gph(d_model, n_heads, granularity):
    d_h = d_model // n_heads
    gate_width(d_model, n_heads, granularity)
    if granularity == "element":
        return d_h, d_h
    return d_h, 1


def project_q_and_content(x, w_qg, b_qg, n_heads, granularity):
    """Widen-project x and split per-head queries from content logits.

    W_qg is (D, H * (d_h + gph)). After the matmul, the last axis is reshaped
    to (H, d_h + gph). Queries are the leading d_h channels; content logits
    are the trailing gph channels. No positions are used.
    """
    d_model = x.shape[-1]
    d_h, gph = _granularity_gph(d_model, n_heads, granularity)
    expected = n_heads * (d_h + gph)
    if w_qg.shape[-1] != expected or b_qg.shape[-1] != expected:
        raise ValueError("W_qg / b_qg width does not match granularity")
    proj = x @ w_qg + b_qg
    proj = jnp.reshape(proj, proj.shape[:-1] + (n_heads, d_h + gph))
    q = proj[..., :d_h]
    c = proj[..., d_h:]
    return q, c


def gate_values(c, bias, n_heads, granularity, d_head=None):
    """Map content logits and a token-shared bias to a head-major gate (..., N, D).

    `bias` is (..., D_g) or a zero scalar. Head-wise gates repeat one sigmoid
    value across d_h channels. Pass `d_head` when `c` still has a trailing
    size of 1 so the flattened gate has width D rather than H.
    """
    if granularity not in ("element", "head"):
        raise ValueError("granularity must be 'element' or 'head'")
    if c.shape[-2] != n_heads:
        raise ValueError("content head axis does not match n_heads")
    bias = jnp.asarray(bias)
    if bias.dtype != c.dtype:
        bias = bias.astype(c.dtype)

    if granularity == "element":
        d_h = int(c.shape[-1])
        if bias.ndim == 0:
            logits = c + bias
        else:
            b = jnp.reshape(bias, bias.shape[:-1] + (1, n_heads, d_h))
            logits = c + b
        g = _sigmoid(logits)
        return jnp.reshape(g, g.shape[:-2] + (n_heads * d_h,))

    width = int(c.shape[-1]) if d_head is None else int(d_head)
    if bias.ndim == 0:
        logits = c[..., :1] + bias
    else:
        b = jnp.reshape(bias, bias.shape[:-1] + (1, n_heads, 1))
        logits = c[..., :1] + b
    sig = _sigmoid(logits)
    lead = sig.shape[:-3]
    n_tok = sig.shape[-3]
    repeated = jnp.broadcast_to(sig, lead + (n_tok, n_heads, width))
    return jnp.reshape(repeated, lead + (n_tok, n_heads * width))


def apply_output_gate(attn, gate, w_o, b_o):
    """Y = (G * A) @ W_o + b_o. Does not renormalize attention."""
    return (gate * attn) @ w_o + b_o


def init_stream_params(key, d_model, n_heads, d_temb, granularity, with_time):
    """Initialize one stream. Content-logit biases are 4; the time branch is 0."""
    d_h, gph = _granularity_gph(d_model, n_heads, granularity)
    d_g = gate_width(d_model, n_heads, granularity)
    k_qg, k_k, k_v, k_o = jax.random.split(key, 4)
    out_qg = n_heads * (d_h + gph)
    w_qg = jax.random.normal(k_qg, (d_model, out_qg), dtype=jnp.float32) * 0.02
    b_qg = jnp.zeros((n_heads, d_h + gph), dtype=jnp.float32)
    b_qg = b_qg.at[:, d_h:].set(4.0)
    b_qg = jnp.reshape(b_qg, (out_qg,))
    params = {
        "w_qg": w_qg,
        "b_qg": b_qg,
        "w_k": jax.random.normal(k_k, (d_model, d_model), dtype=jnp.float32) * 0.02,
        "b_k": jnp.zeros((d_model,), dtype=jnp.float32),
        "w_v": jax.random.normal(k_v, (d_model, d_model), dtype=jnp.float32) * 0.02,
        "b_v": jnp.zeros((d_model,), dtype=jnp.float32),
        "w_o": jax.random.normal(k_o, (d_model, d_model), dtype=jnp.float32) * 0.02,
        "b_o": jnp.zeros((d_model,), dtype=jnp.float32),
        "q_scale": jnp.ones((n_heads, d_h), dtype=jnp.float32),
        "k_scale": jnp.ones((n_heads, d_h), dtype=jnp.float32),
    }
    if with_time:
        params["w_time"] = jnp.zeros((d_temb, d_g), dtype=jnp.float32)
        params["b_time"] = jnp.zeros((d_g,), dtype=jnp.float32)
    return params


def rmsnorm(x, weight, eps=1e-6):
    """Per-head RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * weight."""
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x / jnp.sqrt(ms + eps) * weight


def _align_positions(pos, lead_shape):
    """Broadcast `pos` against leading axes of x, treating the last match as heads."""
    pos = jnp.asarray(pos)
    lead_rank = len(lead_shape)
    if pos.ndim == 0 or pos.ndim == lead_rank:
        return pos
    if pos.ndim > lead_rank:
        raise ValueError("positions rank is incompatible with input")
    last = int(pos.shape[-1])
    order = list(range(lead_rank - 1)) + ([lead_rank - 1] if lead_rank else [])
    target = None
    for ax in order:
        if int(lead_shape[ax]) != last:
            continue
        start = ax - (pos.ndim - 1)
        if start < 0:
            continue
        prefix = tuple(int(s) for s in pos.shape[:-1])
        lead_prefix = tuple(int(lead_shape[i]) for i in range(start, ax))
        if prefix == lead_prefix:
            target = ax
            break
    if target is None:
        target = lead_rank - 1
        if target - (pos.ndim - 1) < 0:
            raise ValueError("positions shape is incompatible with input")
    left_pad = target - pos.ndim + 1
    right_pad = lead_rank - target - 1
    shape = (1,) * left_pad + tuple(int(s) for s in pos.shape) + (1,) * right_pad
    return jnp.reshape(pos, shape)


def rope(x, positions, base=10000.0):
    """Interleaved rotary embedding. Even/odd pairs rotate; the norm is preserved."""
    d_h = int(x.shape[-1])
    if d_h % 2 != 0:
        raise ValueError("RoPE width must be even")
    half = d_h // 2
    dtype = x.dtype
    idx = jnp.arange(half, dtype=dtype)
    inv_freq = jnp.power(
        jnp.asarray(base, dtype=dtype),
        -2.0 * idx / jnp.asarray(d_h, dtype=dtype),
    )
    pos = jnp.asarray(positions)
    if pos.ndim == 0:
        angles = pos.astype(dtype) * inv_freq
    else:
        pos = _align_positions(pos, x.shape[:-1]).astype(dtype)
        angles = pos[..., None] * inv_freq
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    out_even = even * cos - odd * sin
    out_odd = even * sin + odd * cos
    stacked = jnp.stack([out_even, out_odd], axis=-1)
    return jnp.reshape(stacked, x.shape)


def _stream_positions(positions, n_streams):
    if isinstance(positions, (list, tuple)):
        if len(positions) != n_streams:
            raise ValueError("positions must align with streams")
        return list(positions)
    if n_streams == 1:
        return [positions]
    raise ValueError("positions must be a sequence when there are multiple streams")


def joint_attention(streams, params, temb, positions, granularity):
    """Joint attention over one or more streams.

    Returns per-stream outputs, head-averaged pre-gate probabilities, and
    key slices (start, end) in stream order. The softmax is not rescaled by G.
    """
    if len(streams) != len(params):
        raise ValueError("streams and params length mismatch")
    if len(streams) == 0:
        raise ValueError("at least one stream is required")
    positions = _stream_positions(positions, len(streams))
    x0 = streams[0]
    d_model = int(x0.shape[-1])
    n_heads = int(params[0]["q_scale"].shape[0])
    d_h, _gph = _granularity_gph(d_model, n_heads, granularity)
    scale = jnp.sqrt(jnp.asarray(d_h, dtype=x0.dtype))

    qs = []
    ks = []
    vs = []
    gates = []
    key_slices = []
    cursor = 0
    for x, p, pos in zip(streams, params, positions):
        q, c = project_q_and_content(x, p["w_qg"], p["b_qg"], n_heads, granularity)
        k = x @ p["w_k"] + p["b_k"]
        v = x @ p["w_v"] + p["b_v"]
        k = jnp.reshape(k, k.shape[:-1] + (n_heads, d_h))
        v = jnp.reshape(v, v.shape[:-1] + (n_heads, d_h))
        q = rmsnorm(q, p["q_scale"])
        k = rmsnorm(k, p["k_scale"])
        q = rope(q, pos)
        k = rope(k, pos)
        if "w_time" in p:
            if temb is None:
                raise ValueError("temb is required when a stream has a time branch")
            bias = time_bias(temb, p["w_time"], p["b_time"])
        else:
            bias = jnp.asarray(0.0, dtype=x.dtype)
        g = gate_values(c, bias, n_heads, granularity, d_head=d_h)
        qs.append(q)
        ks.append(k)
        vs.append(v)
        gates.append(g)
        n_tok = int(x.shape[1])
        key_slices.append((cursor, cursor + n_tok))
        cursor += n_tok

    keys = jnp.concatenate(ks, axis=1)
    values = jnp.concatenate(vs, axis=1)
    outputs = []
    probs = []
    for q, g, p in zip(qs, gates, params):
        logits = jnp.einsum("bnhd,bmhd->bhnm", q, keys) / scale
        prob = jax.nn.softmax(logits, axis=-1)
        attn = jnp.einsum("bhnm,bmhd->bnhd", prob, values)
        attn = jnp.reshape(attn, attn.shape[:-2] + (d_model,))
        outputs.append(apply_output_gate(attn, g, p["w_o"], p["b_o"]))
        probs.append(jnp.mean(prob, axis=1))
    return outputs, probs, key_slices


def image_to_text_fraction(probs, text_key_slice):
    """Mean over image queries of attention mass on a half-open text key slice."""
    start, end = text_key_slice
    mass = jnp.sum(probs[..., start:end], axis=-1)
    return jnp.mean(mass, axis=-1)


def peak_attention_mass(probs):
    """max over keys of the mean attention mass across image queries."""
    return jnp.max(jnp.mean(probs, axis=-2), axis=-1)
