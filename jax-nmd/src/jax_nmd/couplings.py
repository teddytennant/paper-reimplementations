"""AmpNet left/right heads: per-port complex couplings (appendix B.3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_nmd.config import NMDConfig
from jax_nmd.poles import _init_linear, attn_block, init_attn_block


def complex_from_mag_phase(raw_mag, phase):
    """a = softplus(raw_mag) * exp(j phase). Magnitude is strictly positive."""
    mag = jax.nn.softplus(raw_mag)
    return jax.lax.complex(mag * jnp.cos(phase), mag * jnp.sin(phase))


def init_amp_net(key, cfg: NMDConfig):
    k_port, k_attn, k_out = jax.random.split(key, 3)
    d_t, k_poles = cfg.d_t, cfg.k_poles
    port_w, port_b = _init_linear(k_port, cfg.port_dim, d_t)
    out_w, out_b = _init_linear(k_out, d_t, 2 * k_poles, scale=0.05)
    return {
        "port_w": port_w,
        "port_b": port_b,
        "attn": init_attn_block(k_attn, d_t, cfg.hidden),
        "out_w": out_w,
        "out_b": out_b,
    }


def amp_net_one_port(params, tokens, port_loc, cfg: NMDConfig):
    """tokens (B, n_tokens, D), port_loc (B, port_dim) -> (B, K) complex."""
    q = jax.nn.gelu(port_loc @ params["port_w"] + params["port_b"])
    q = q[:, None, :]
    x = attn_block(params["attn"], q, tokens, ln_eps=cfg.ln_eps)
    x = x[:, 0, :]
    raw = x @ params["out_w"] + params["out_b"]
    raw = raw.reshape(raw.shape[0], cfg.k_poles, 2)
    return complex_from_mag_phase(raw[..., 0], raw[..., 1])


def amp_net_ports(params, tokens, port_locs, cfg: NMDConfig):
    """Query each port independently.

    port_locs: (B, N, port_dim) -> couplings (B, K, N) complex.
    """

    def one(port_loc):
        return amp_net_one_port(params, tokens, port_loc, cfg)

    return jax.vmap(one, in_axes=1, out_axes=-1)(port_locs)
