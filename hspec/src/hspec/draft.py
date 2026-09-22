"""Tiny causal target and a mask-prompted drafter with hybrid injection.

The drafter does not store a prefix KV cache. Target K/V are arguments to
attention; only the draft block is projected to Q, K, and V. RoPE positions
on draft Q/K continue from the prefix. Gamma proposals come from one shared
mask embedding (no positional embedding on that input). The selective scan
is started from ``s_init`` of the target's last-step hiddens.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hspec.inject import s_init, scan_parallel

D = 32
VOCAB = 32
N_LAYERS = 2
N_HEADS = 4
D_HEAD = D // N_HEADS
D_STATE = 32
D_FF = 64
DEFAULT_GAMMA = 4


def apply_rope(x, positions):
    """Rotate the last axis of ``x``. ``positions`` has one entry per row.

    ``x`` is ``(seq, d)`` or ``(seq, heads, d_head)``. Position ``p`` uses
    angles ``p / 10000^{2i/d}``, so a nonempty prefix must not pass 0 for
    the first block token.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    positions = jnp.asarray(positions)
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (jnp.arange(half, dtype=jnp.float32) / half))
    ang = positions.astype(jnp.float32)[:, None] * freq[None, :]
    cos = jnp.cos(ang)
    sin = jnp.sin(ang)
    bshape = (x.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    cos = cos.reshape(bshape)
    sin = sin.reshape(bshape)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def block_positions(prefix_len, gamma):
    """Absolute positions of a draft block: ``prefix_len + i``."""
    return jnp.asarray(prefix_len, dtype=jnp.int32) + jnp.arange(gamma, dtype=jnp.int32)


def _xavier(key, shape):
    fan = (shape[0] + shape[-1]) / 2.0
    return jax.random.normal(key, shape, dtype=jnp.float32) * jnp.sqrt(2.0 / fan)


def init_target(key):
    """2-layer causal transformer, d=32, vocab=32."""
    keys = jax.random.split(key, 2 + N_LAYERS * 6)
    params = {
        "embed": jax.random.normal(keys[0], (VOCAB, D), dtype=jnp.float32) * 0.1,
        "ln_f": jnp.ones((D,), dtype=jnp.float32),
        "lm_head": _xavier(keys[1], (D, VOCAB)) * 0.5,
        "layers": [],
    }
    ki = 2
    for _ in range(N_LAYERS):
        params["layers"].append(
            {
                "ln1": jnp.ones((D,), dtype=jnp.float32),
                "ln2": jnp.ones((D,), dtype=jnp.float32),
                "W_q": _xavier(keys[ki], (D, N_HEADS, D_HEAD)),
                "W_k": _xavier(keys[ki + 1], (D, N_HEADS, D_HEAD)),
                "W_v": _xavier(keys[ki + 2], (D, N_HEADS, D_HEAD)),
                "W_o": _xavier(keys[ki + 3], (N_HEADS, D_HEAD, D)),
                "W_up": _xavier(keys[ki + 4], (D, D_FF)),
                "W_down": _xavier(keys[ki + 5], (D_FF, D)),
            }
        )
        ki += 6
    return params


def _layer_norm(x, scale, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return scale * (x - mean) * jnp.reciprocal(jnp.sqrt(var + eps))


def _causal_attend(q, k, v):
    """q, k, v: (batch, seq, heads, d_head) -> (batch, seq, heads, d_head)."""
    scale = D_HEAD ** -0.5
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    seq = q.shape[1]
    idx = jnp.arange(seq)
    allow = idx[None, :] <= idx[:, None]
    scores = jnp.where(allow[None, None, :, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def target_forward(params, tokens):
    """Batched causal forward.

    ``tokens`` is ``(batch, seq)``. Returns ``logits (batch, seq, vocab)``,
    a list of per-layer hiddens ``(batch, seq, d)``, and a list of
    ``(k, v)`` with post-RoPE K and raw V, each ``(batch, seq, heads, d_head)``.
    """
    tokens = jnp.asarray(tokens)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    batch, seq = tokens.shape
    x = params["embed"][tokens]
    positions = jnp.arange(seq, dtype=jnp.int32)
    hiddens = []
    kvs = []
    for layer in params["layers"]:
        h = _layer_norm(x, layer["ln1"])
        q = jnp.einsum("bsd,dhe->bshe", h, layer["W_q"])
        k = jnp.einsum("bsd,dhe->bshe", h, layer["W_k"])
        v = jnp.einsum("bsd,dhe->bshe", h, layer["W_v"])
        q = apply_rope(q.reshape(batch * seq, N_HEADS, D_HEAD), jnp.tile(positions, (batch,)))
        k = apply_rope(k.reshape(batch * seq, N_HEADS, D_HEAD), jnp.tile(positions, (batch,)))
        q = q.reshape(batch, seq, N_HEADS, D_HEAD)
        k = k.reshape(batch, seq, N_HEADS, D_HEAD)
        kvs.append((k, v))
        attn = _causal_attend(q, k, v)
        attn = jnp.einsum("bshe,hed->bsd", attn, layer["W_o"])
        x = x + attn
        h = _layer_norm(x, layer["ln2"])
        h = jax.nn.gelu(h @ layer["W_up"]) @ layer["W_down"]
        x = x + h
        hiddens.append(x)
    final = _layer_norm(x, params["ln_f"])
    logits = final @ params["lm_head"]
    return logits, hiddens, kvs


def init_drafter(key):
    """Drafter parameters. Independent of prefix length.

    One ``proj_r`` and one ``proj_ds`` (shared across target layers).
    One shared ``mask_embed`` vector, not a position table.
    """
    keys = jax.random.split(key, 16)
    proj_r = jnp.zeros((D_STATE, N_LAYERS * D), dtype=jnp.float32)
    proj_r = proj_r.at[:, D:].set(jnp.eye(D_STATE) * 0.05)
    feat = D_STATE + D_STATE + D + 8
    return {
        "mask_embed": jax.random.normal(keys[0], (D,), dtype=jnp.float32) * 0.5,
        "W_q": _xavier(keys[1], (D, N_HEADS, D_HEAD)),
        "W_k": _xavier(keys[2], (D, N_HEADS, D_HEAD)),
        "W_v": _xavier(keys[3], (D, N_HEADS, D_HEAD)),
        "W_o": _xavier(keys[4], (N_HEADS, D_HEAD, D)),
        "W_ctx": _xavier(keys[5], (D_STATE, D)) * 0.1,
        "proj_r": proj_r,
        "proj_ds": jnp.eye(D_STATE, dtype=jnp.float32),
        "A": jnp.full((D_STATE,), -4.0, dtype=jnp.float32),
        "W_B": _xavier(keys[6], (D_STATE, D)) * 0.05,
        "W_dt": _xavier(keys[7], (D, D_STATE)) * 0.05,
        "W_x": _xavier(keys[8], (D, D_STATE)) * 0.1,
        "W_mlp": _xavier(keys[9], (feat, D)) * 0.5,
        "lm_head": _xavier(keys[10], (D, VOCAB)) * 0.2,
        "W_state_logit": _xavier(keys[11], (D_STATE, VOCAB)) * 0.05,
    }


def count_params(params):
    return int(sum(int(jnp.size(leaf)) for leaf in jax.tree_util.tree_leaves(params)))


def mask_inputs(params, gamma):
    """Shared mask embedding repeated across the block. No position add."""
    return jnp.repeat(params["mask_embed"][None, :], gamma, axis=0)


def draft_qkv(params, gamma, prefix_len):
    """Draft Q, K, V for the block only.

    Q and K are RoPE'd at ``prefix_len + i``. Nothing prefix-sized is returned.
    """
    x = mask_inputs(params, gamma)
    q = jnp.einsum("gd,dhe->ghe", x, params["W_q"])
    k = jnp.einsum("gd,dhe->ghe", x, params["W_k"])
    v = jnp.einsum("gd,dhe->ghe", x, params["W_v"])
    pos = block_positions(prefix_len, gamma)
    q = apply_rope(q, pos)
    k = apply_rope(k, pos)
    return q, k, v


def draft_attention(q, k, v, target_k, target_v):
    """Attend over ``concat(target_k, draft_k)``.

    Prefix keys are fully visible. The block is causal. Returns only the
    block output ``(gamma, heads, d_head)`` — no cache.
    """
    target_k = jnp.asarray(target_k, dtype=jnp.float32)
    target_v = jnp.asarray(target_v, dtype=jnp.float32)
    prefix = target_k.shape[0]
    if prefix == 0:
        keys = k
        vals = v
    else:
        keys = jnp.concatenate([target_k, k], axis=0)
        vals = jnp.concatenate([target_v, v], axis=0)
    gamma = q.shape[0]
    scale = D_HEAD ** -0.5
    scores = jnp.einsum("ghd,shd->hgs", q, keys) * scale
    key_idx = jnp.arange(prefix + gamma)
    query_idx = jnp.arange(gamma)
    allow = key_idx[None, :] <= (prefix + query_idx)[:, None]
    scores = jnp.where(allow[None, :, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hgs,shd->ghd", weights, vals)


def _zeros_kv_like(target_k):
    tail = target_k.shape[1:]
    return jnp.zeros((0,) + tail, dtype=jnp.float32)


def _rms(v, eps=1e-5):
    v = jnp.asarray(v, dtype=jnp.float32)
    return v / jnp.sqrt(jnp.mean(v * v) + eps)


def _block_pos_features(prefix_len, gamma):
    """Sinusoids of absolute block positions. Not added to the mask input."""
    pos = block_positions(prefix_len, gamma).astype(jnp.float32)[:, None]
    freqs = jnp.arange(4, dtype=jnp.float32)
    ang = pos / (10000 ** (freqs / 4.0))
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def drafter_forward(
    params,
    h_last,
    target_k,
    target_v,
    prefix_len,
    gamma=DEFAULT_GAMMA,
    inject_state=True,
    inject_kv=True,
):
    """Logits ``(gamma, vocab)`` for one example.

    ``h_last`` is passed to ``s_init`` (last-step target hiddens).
    ``target_k`` / ``target_v`` are arguments, not stored. With
    ``inject_state=False`` the scan starts at 0. With ``inject_kv=False``
    the prefix is omitted from attention.
    """
    gamma = int(gamma)
    if inject_state:
        state0 = s_init(h_last, params["proj_r"], params["proj_ds"])
        state0 = _rms(state0)
    else:
        state0 = jnp.zeros((D_STATE,), dtype=jnp.float32)
    if not inject_kv:
        target_k = _zeros_kv_like(jnp.asarray(target_k))
        target_v = _zeros_kv_like(jnp.asarray(target_v))
    else:
        target_k = jnp.asarray(target_k, dtype=jnp.float32)
        target_v = jnp.asarray(target_v, dtype=jnp.float32)

    x = mask_inputs(params, gamma)
    ctx = state0 @ params["W_ctx"]
    q_in = x + ctx[None, :]
    q = jnp.einsum("gd,dhe->ghe", q_in, params["W_q"])
    k = jnp.einsum("gd,dhe->ghe", x, params["W_k"])
    v = jnp.einsum("gd,dhe->ghe", x, params["W_v"])
    pos = block_positions(prefix_len, gamma)
    q = apply_rope(q, pos)
    k = apply_rope(k, pos)
    attn_h = draft_attention(q, k, v, target_k, target_v)
    attn = jnp.einsum("ghe,hed->gd", attn_h, params["W_o"])
    attn = jax.vmap(_rms)(attn)
    h = x + attn
    dt = h @ params["W_dt"]
    x_proj = h @ params["W_x"]
    states = scan_parallel(h, dt, params["A"], params["W_B"], x_proj, state0)
    states = jax.vmap(_rms)(states)
    state_rep = jnp.repeat(state0[None, :], gamma, axis=0)
    feat = jnp.concatenate([states, state_rep, attn, _block_pos_features(prefix_len, gamma)], axis=-1)
    hidden = jax.nn.gelu(feat @ params["W_mlp"])
    return hidden @ params["lm_head"] + state_rep @ params["W_state_logit"]


def drafter_forward_batch(
    params,
    h_last,
    target_k,
    target_v,
    prefix_len,
    gamma=DEFAULT_GAMMA,
    inject_state=True,
    inject_kv=True,
):
    """Batched wrapper. ``h_last`` is ``(batch, n_layers, d)``."""

    def one(h, tk, tv):
        return drafter_forward(
            params,
            h,
            tk,
            tv,
            prefix_len,
            gamma,
            inject_state=inject_state,
            inject_kv=inject_kv,
        )

    return jax.vmap(one)(h_last, target_k, target_v)


def propose(params, h_last, target_k, target_v, prefix_len, gamma=DEFAULT_GAMMA, inject_state=True, inject_kv=True):
    logits = drafter_forward(
        params,
        h_last,
        target_k,
        target_v,
        prefix_len,
        gamma,
        inject_state=inject_state,
        inject_kv=inject_kv,
    )
    return jnp.argmax(logits, axis=-1)


def acceptance_length(draft_tokens, target_tokens):
    """Leading matches against the target's tokens. Mismatch at j -> length j."""
    draft_tokens = jnp.asarray(draft_tokens)
    target_tokens = jnp.asarray(target_tokens)
    match = (draft_tokens == target_tokens).astype(jnp.int32)
    return jnp.sum(jnp.cumprod(match, axis=-1), axis=-1)


def accepted_prefix(draft_tokens, target_tokens):
    """Accepted tokens. Equal to the target's own tokens on that prefix."""
    length = int(acceptance_length(draft_tokens, target_tokens))
    return jnp.asarray(target_tokens)[:length]


def target_greedy_tokens(params, prefix, gamma):
    """Target's own next ``gamma`` tokens (greedy), prefix is ``(seq,)`` or ``(batch, seq)``."""
    prefix = jnp.asarray(prefix)
    squeeze = prefix.ndim == 1
    if squeeze:
        prefix = prefix[None, :]
    tokens = prefix
    preds = []
    for _ in range(int(gamma)):
        logits, _, _ = target_forward(params, tokens)
        nxt = jnp.argmax(logits[:, -1, :], axis=-1)
        preds.append(nxt)
        tokens = jnp.concatenate([tokens, nxt[:, None]], axis=1)
    out = jnp.stack(preds, axis=1)
    return out[0] if squeeze else out


def last_hiddens(hiddens, index):
    """Stack per-layer hiddens at one time index -> ``(batch, n_layers, d)``."""
    return jnp.stack([h[:, index, :] for h in hiddens], axis=1)


def sample_motif_batch(key, batch, seq_len, motif_len=4):
    """Repeating motif with a per-sequence offset and phase.

    Next tokens are a function of the prefix contents, not of absolute position.
    """
    k1, k2 = jax.random.split(key)
    offset = jax.random.randint(k1, (batch, 1), 0, VOCAB)
    phase = jax.random.randint(k2, (batch, 1), 0, motif_len)
    t = jnp.arange(seq_len)[None, :]
    return (offset + (t + phase) % motif_len) % VOCAB
