"""Init, forward, synthetic LTI batches, and AdamW (lr 3e-4, wd 1e-3)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from jax_nmd.cavity import cavity_encode, init_cavity_net
from jax_nmd.config import NMDConfig
from jax_nmd.couplings import amp_net_ports, init_amp_net
from jax_nmd.loss import nmd_total_loss
from jax_nmd.poles import init_pole_net, pole_net, poles_from_raw
from jax_nmd.synthesize import synthesize_z, z_to_s


def init_params(key, cfg: NMDConfig):
    k_c, k_p, k_l, k_r = jax.random.split(key, 4)
    return {
        "cavity": init_cavity_net(k_c, cfg),
        "pole": init_pole_net(k_p, cfg),
        "amp_left": init_amp_net(k_l, cfg),
        "amp_right": init_amp_net(k_r, cfg),
    }


def forward(params, geometry, port_locs, omega, cfg: NMDConfig):
    """Poles from geometry, couplings from (geometry, port), then Z and S.

    geometry: (B, H, W)
    port_locs: (B, N, port_dim)
    omega: (F,)
    """
    tokens = cavity_encode(params["cavity"], geometry, cfg)
    poles = pole_net(params["pole"], tokens, cfg)
    left = amp_net_ports(params["amp_left"], tokens, port_locs, cfg)
    right = amp_net_ports(params["amp_right"], tokens, port_locs, cfg)
    z = synthesize_z(omega, poles, left, right)
    s = z_to_s(z, z0=cfg.z0)
    return {"tokens": tokens, "poles": poles, "left": left, "right": right, "Z": z, "S": s}


def predict_loss(params, batch, cfg: NMDConfig):
    out = forward(params, batch["geometry"], batch["port_locs"], batch["omega"], cfg)
    loss, parts = nmd_total_loss(
        out["S"],
        batch["S"],
        out["Z"],
        batch["Z"],
        out["left"],
        out["right"],
        batch["is_anchor"],
        cfg,
    )
    return loss, {**parts, **out}


def make_synthetic_batch(
    key,
    cfg: NMDConfig,
    batch_size=2,
    n_ports=2,
    n_freq=16,
    k_true=None,
):
    """Sample LTI poles/couplings, build Z_gt by the same pole-residue sum."""
    k_true = cfg.k_poles if k_true is None else k_true
    k_u, k_v, k_lm, k_lp, k_rm, k_rp, k_g, k_port = jax.random.split(key, 8)
    u = jax.random.normal(k_u, (batch_size, k_true)) - 0.5
    v = 2.0 + 6.0 * jax.random.uniform(k_v, (batch_size, k_true))
    poles = poles_from_raw(u, v)
    mag_l = jax.nn.softplus(jax.random.normal(k_lm, (batch_size, k_true, n_ports)))
    mag_r = jax.nn.softplus(jax.random.normal(k_rm, (batch_size, k_true, n_ports)))
    ph_l = jax.random.uniform(k_lp, (batch_size, k_true, n_ports), minval=-jnp.pi, maxval=jnp.pi)
    ph_r = jax.random.uniform(k_rp, (batch_size, k_true, n_ports), minval=-jnp.pi, maxval=jnp.pi)
    left = jax.lax.complex(mag_l * jnp.cos(ph_l), mag_l * jnp.sin(ph_l))
    right = jax.lax.complex(mag_r * jnp.cos(ph_r), mag_r * jnp.sin(ph_r))
    omega = jnp.linspace(1.0, 10.0, n_freq, dtype=jnp.float32)
    z = synthesize_z(omega, poles, left, right)
    s = z_to_s(z, z0=cfg.z0)
    geometry = jax.random.bernoulli(
        k_g, 0.5, (batch_size, cfg.geom_h, cfg.geom_w)
    ).astype(jnp.float32)
    port_locs = jax.random.uniform(k_port, (batch_size, n_ports, cfg.port_dim))
    is_anchor = port_locs[..., 0] > 0.5
    return {
        "geometry": geometry,
        "port_locs": port_locs,
        "omega": omega,
        "Z": z,
        "S": s,
        "is_anchor": is_anchor,
        "poles": poles,
        "left": left,
        "right": right,
    }


def init_adamw(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adamw_update(params, grads, opt, lr=3e-4, wd=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = opt["t"] + 1
    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, opt["m"], grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * g * g, opt["v"], grads)
    mhat = jax.tree_util.tree_map(lambda mi: mi / (1.0 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vi: vi / (1.0 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + wd * p),
        params,
        mhat,
        vhat,
    )
    return params, {"m": m, "v": v, "t": t}


@partial(jax.jit, static_argnames=("cfg",))
def train_step(params, opt, batch, cfg: NMDConfig):
    def loss_only(p):
        loss, _ = predict_loss(p, batch, cfg)
        return loss

    loss, grads = jax.value_and_grad(loss_only)(params)
    params, opt = adamw_update(
        params, grads, opt, lr=cfg.lr, wd=cfg.weight_decay
    )
    return params, opt, loss
