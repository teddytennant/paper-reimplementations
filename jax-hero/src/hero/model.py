"""Tiny frozen decoder with token-wise HeRo FFN routing on every layer."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from hero.config import HeroConfig
from hero.ops import layer_norm, linear, routed_residual, skip_loss
from hero.router import init_shared_router, route_block

Params = dict[str, Any]


def _n(key: jax.Array, shape: tuple[int, ...], scale: float = 0.1) -> jax.Array:
    return scale * jax.random.normal(key, shape, dtype=jnp.float32)


def init_backbone_layer(key: jax.Array, cfg: HeroConfig) -> dict[str, jax.Array]:
    keys = jax.random.split(key, 8)
    return {
        "ln_attn_scale": jnp.ones((cfg.d_model,), dtype=jnp.float32),
        "ln_attn_bias": jnp.zeros((cfg.d_model,), dtype=jnp.float32),
        "W_q": _n(keys[0], (cfg.d_model, cfg.d_model)),
        "W_k": _n(keys[1], (cfg.d_model, cfg.d_model)),
        "W_v": _n(keys[2], (cfg.d_model, cfg.d_model)),
        "W_o": _n(keys[3], (cfg.d_model, cfg.d_model)),
        "ln_ffn_scale": jnp.ones((cfg.d_model,), dtype=jnp.float32),
        "ln_ffn_bias": jnp.zeros((cfg.d_model,), dtype=jnp.float32),
        "W_ff1": _n(keys[4], (cfg.d_ff, cfg.d_model)),
        "W_ff2": _n(keys[5], (cfg.d_model, cfg.d_ff), scale=0.05),
    }


def init_adapter(key: jax.Array, cfg: HeroConfig) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)
    return {
        "W1": _n(k1, (cfg.d_r, cfg.d_model)),
        "W2": _n(k2, (cfg.d_model, cfg.d_r), scale=0.05),
    }


def init_params(key: jax.Array, cfg: HeroConfig) -> Params:
    keys = jax.random.split(key, 4 + cfg.n_layers)
    embed = _n(keys[0], (cfg.vocab_size, cfg.d_model), scale=0.02)
    lm_head = _n(keys[1], (cfg.vocab_size, cfg.d_model), scale=0.02)
    router = init_shared_router(keys[2], cfg)
    layers = []
    for i in range(cfg.n_layers):
        lk = jax.random.split(keys[3 + i], 2)
        layers.append(
            {
                "backbone": init_backbone_layer(lk[0], cfg),
                "adapter": init_adapter(lk[1], cfg),
                "eta": jnp.zeros((), dtype=jnp.float32),
            }
        )
    return {
        "embed": embed,
        "lm_head": lm_head,
        "router": router,
        "layers": layers,
    }


def _split_heads(x: jax.Array, n_heads: int) -> jax.Array:
    batch, seq, dim = x.shape
    d_head = dim // n_heads
    return x.reshape(batch, seq, n_heads, d_head).transpose(0, 2, 1, 3)


def _merge_heads(x: jax.Array) -> jax.Array:
    batch, n_heads, seq, d_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq, n_heads * d_head)


def attention(x: jax.Array, layer: dict[str, jax.Array], cfg: HeroConfig) -> jax.Array:
    q = _split_heads(linear(x, layer["W_q"]), cfg.n_heads)
    k = _split_heads(linear(x, layer["W_k"]), cfg.n_heads)
    v = _split_heads(linear(x, layer["W_v"]), cfg.n_heads)
    scale = jnp.sqrt(jnp.asarray(cfg.d_head, dtype=x.dtype))
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / scale
    seq = x.shape[1]
    causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
    scores = jnp.where(causal, scores, jnp.full_like(scores, -1e9))
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    return linear(_merge_heads(out), layer["W_o"])


def frozen_ffn(u: jax.Array, layer: dict[str, jax.Array]) -> jax.Array:
    return linear(jax.nn.silu(linear(u, layer["W_ff1"])), layer["W_ff2"])


def adapter_ffn(u: jax.Array, adapter: dict[str, jax.Array]) -> jax.Array:
    """Adapter(u) = W2 SiLU(W1 u) (eq. 2)."""
    return linear(jax.nn.silu(linear(u, adapter["W1"])), adapter["W2"])


def forward(
    params: Params,
    tokens: jax.Array,
    cfg: HeroConfig,
    *,
    return_aux: bool = False,
) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    """Run the frozen stack with HeRo FFN routing on every layer.

    Attention stays dense (eq. 1). FFN vs adapter is chosen per token (eq. 3).
    """
    h = params["embed"][tokens]
    n_routed = cfg.n_routed
    batch, seq, _ = h.shape
    d_m = cfg.d_m
    s_mem = jnp.zeros((batch, seq, d_m, d_m), dtype=h.dtype)
    zeta = jnp.zeros((batch, seq, d_m), dtype=h.dtype)
    p_hist_prev = jnp.ones((batch, seq), dtype=h.dtype)
    m_hist_prev = jnp.ones((batch, seq), dtype=h.dtype)
    hbar_prev: jax.Array | None = None
    delta_prev: jax.Array | None = None

    gates = []
    contexts = []
    nus = []
    p_hists = []
    m_hists = []
    a_list = []
    b_list = []
    s_after = []
    zeta_after = []
    rhos = []
    phi_ks = []
    vs = []

    for i, layer in enumerate(params["layers"]):
        bb = layer["backbone"]
        attn_in = layer_norm(h, bb["ln_attn_scale"], bb["ln_attn_bias"], cfg.ln_eps)
        hbar = h + attention(attn_in, bb, cfg)
        u = layer_norm(hbar, bb["ln_ffn_scale"], bb["ln_ffn_bias"], cfg.ln_eps)
        ffn_out = frozen_ffn(u, bb)
        adapter_out = adapter_ffn(u, layer["adapter"])

        j = i + 1
        step = route_block(
            hbar,
            j,
            n_routed,
            params["router"],
            layer["eta"],
            s_mem,
            zeta,
            hbar_prev,
            delta_prev,
            p_hist_prev,
            m_hist_prev,
            cfg,
        )
        h = routed_residual(hbar, step["g"], ffn_out, adapter_out, cfg.tau_exec)

        s_mem = step["S"]
        zeta = step["zeta"]
        p_hist_prev = step["p_hist"]
        m_hist_prev = step["m_hist"]
        delta_prev = step["delta"]
        hbar_prev = hbar

        gates.append(step["g"])
        contexts.append(step["c"])
        nus.append(step["nu"])
        p_hists.append(step["p_hist"])
        m_hists.append(step["m_hist"])
        a_list.append(step["a"])
        b_list.append(step["b"])
        s_after.append(step["S"])
        zeta_after.append(step["zeta"])
        rhos.append(jnp.broadcast_to(step["rho"], step["g"].shape))
        phi_ks.append(step["phi_k"])
        vs.append(step["v"])

    logits = linear(h, params["lm_head"])
    if not return_aux:
        return logits

    aux = {
        "logits": logits,
        "h": h,
        "gates": jnp.stack(gates, axis=-1),
        "c": jnp.stack(contexts, axis=-1),
        "nu": jnp.stack(nus, axis=-1),
        "p_hist": jnp.stack(p_hists, axis=-1),
        "m_hist": jnp.stack(m_hists, axis=-1),
        "a": jnp.stack(a_list, axis=-1),
        "b": jnp.stack(b_list, axis=-1),
        "S": jnp.stack(s_after, axis=2),
        "zeta": jnp.stack(zeta_after, axis=2),
        "rho": jnp.stack(rhos, axis=-1),
        "phi_k": jnp.stack(phi_ks, axis=2),
        "v": jnp.stack(vs, axis=2),
    }
    return logits, aux


def language_model_loss(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    pred = logits[:, :-1]
    target = tokens[:, 1:]
    log_probs = jax.nn.log_softmax(pred, axis=-1)
    nll = -jnp.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]
    return jnp.mean(nll)


def total_loss(
    params: Params, tokens: jax.Array, cfg: HeroConfig
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """L = L_LM + alpha_t L_skip (eq. 17)."""
    logits, aux = forward(params, tokens, cfg, return_aux=True)
    lm = language_model_loss(logits, tokens)
    skip = skip_loss(aux["gates"][:, :-1, :])
    loss = lm + cfg.alpha_t * skip
    aux = {**aux, "lm": lm, "skip": skip, "loss": loss}
    return loss, aux
