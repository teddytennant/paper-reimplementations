"""Tiny conv encoder/decoder used only in tests and smoke. Not FSNet/ALGNet."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp

from jax_gralod.align import align_pyramid, map_pyramid_to_stage
from jax_gralod.estimator import apply_probe, inject_stage
from jax_gralod.losses import (
    calibration_ramp,
    l_cal,
    l_probe,
    l_str,
    msfc_target,
    reconstruction_loss,
    resize_clean,
    stack_level_errors,
    total_loss,
)
from jax_gralod.model import GraLoDConfig, init as init_gralod
from jax_gralod.ops import Array, Params, antialiased_resize, conv1x1, conv3x3, init_conv2d, relu


def init_backbone(key: Array, cfg: GraLoDConfig) -> Params:
    n = cfg.n_levels
    n_fuse = max(n - 1, 0)
    keys = jax.random.split(key, n + n_fuse + 2)
    enc = []
    in_ch = cfg.in_channels
    for l, ch in enumerate(cfg.encoder_channels):
        enc.append(init_conv2d(keys[l], 3, 3, in_ch, ch))
        in_ch = ch
    dec = []
    # Coarse-to-fine fuse: for s = n-2 .. 0, concat(up(D_{s+1}), F^s) -> D_s
    for i, s in enumerate(range(n - 2, -1, -1)):
        in_fuse = cfg.decoder_channels[s + 1] + cfg.encoder_channels[s]
        dec.append(init_conv2d(keys[n + i], 3, 3, in_fuse, cfg.decoder_channels[s]))
    head = init_conv2d(keys[n + n_fuse], 1, 1, cfg.decoder_channels[0], cfg.out_channels)
    bottleneck = init_conv2d(
        keys[n + n_fuse + 1], 1, 1, cfg.encoder_channels[-1], cfg.decoder_channels[-1]
    )
    return {"enc": enc, "dec": dec, "head": head, "bottleneck": bottleneck}


def encode(params: Params, image: Array) -> list[Array]:
    """F^l at H, H/2, ... (stride 1 then stride 2 downs). Fine to coarse."""
    feats = []
    h = image
    for i, conv_p in enumerate(params["enc"]):
        stride = 1 if i == 0 else 2
        h = relu(conv3x3(conv_p, h, stride=stride))
        feats.append(h)
    return feats


def init_restorer(key: Array, cfg: GraLoDConfig) -> Params:
    k1, k2 = jax.random.split(key)
    return {"backbone": init_backbone(k1, cfg), "gralod": init_gralod(k2, cfg)}


def _decode_with_inject(
    bb: Params,
    gralod: Params,
    feats: Sequence[Array],
    cfg: GraLoDConfig,
    with_probes: bool,
) -> tuple[Array, dict[str, Any]]:
    n = cfg.n_levels
    z = align_pyramid(gralod["align"], feats, ln_eps=cfg.ln_eps)
    d = conv1x1(bb["bottleneck"], feats[-1])
    lambdas: list[Array | None] = [None] * n
    z_stages: list[tuple[Array, ...] | None] = [None] * n
    d_list: list[Array | None] = [None] * n
    probe_imgs: list[tuple[Array, ...] | None] = [None] * n

    def run_stage(s: int, d_s: Array) -> Array:
        _, h, w, _ = d_s.shape
        z_s = map_pyramid_to_stage(z, h, w)
        d_t, lam, _f = inject_stage(gralod["stages"][s], d_s, z_s, n)
        lambdas[s] = lam
        z_stages[s] = z_s
        d_list[s] = d_s
        if with_probes:
            probe_imgs[s] = tuple(
                apply_probe(gralod["stages"][s]["P"], d_s, z_l, detach_inputs=True)
                for z_l in z_s
            )
        return d_t

    d = run_stage(n - 1, d)
    # decoder fuses are stored fine-ward: index 0 is fuse into stage n-2, last into stage 0
    for i, s in enumerate(range(n - 2, -1, -1)):
        _, h, w, _ = feats[s].shape
        d = antialiased_resize(d, h, w)
        fused = jnp.concatenate([d, feats[s]], axis=-1)
        d = relu(conv3x3(bb["dec"][i], fused))
        d = run_stage(s, d)

    rgb = conv1x1(bb["head"], d)
    aux = {
        "lambdas": list(lambdas),
        "z_stages": list(z_stages),
        "d_list": list(d_list),
        "probes": list(probe_imgs) if with_probes else None,
        "feats": feats,
    }
    return rgb, aux


def forward_infer(params: Params, degraded: Array, cfg: GraLoDConfig) -> Array:
    feats = encode(params["backbone"], degraded)
    rgb, _aux = _decode_with_inject(
        params["backbone"], params["gralod"], feats, cfg, with_probes=False
    )
    return rgb


def forward_train(
    params: Params,
    degraded: Array,
    clean: Array,
    cfg: GraLoDConfig,
    step: int = 0,
) -> tuple[Array, dict[str, Any]]:
    feats = encode(params["backbone"], degraded)
    rgb, aux = _decode_with_inject(
        params["backbone"], params["gralod"], feats, cfg, with_probes=True
    )
    x_stages = []
    targets = []
    for s, d_s in enumerate(aux["d_list"]):
        _, h, w, _ = d_s.shape
        x_s = resize_clean(clean, h, w)
        x_stages.append(x_s)
        errors = stack_level_errors(aux["probes"][s], x_s, ksize=cfg.box_ksize)
        lam_star, _q, _c = msfc_target(errors, gamma=cfg.gamma, tau=cfg.tau)
        targets.append(lam_star)

    rec = reconstruction_loss(rgb, clean)
    cal = l_cal(aux["lambdas"], targets)
    str_loss = l_str(aux["lambdas"], x_stages, beta=cfg.beta)
    probe = l_probe(aux["probes"], x_stages)
    lam_cal = calibration_ramp(step, cfg.warmup_steps, cfg.lambda_cal)
    loss = total_loss(
        rec,
        cal,
        str_loss,
        probe,
        lambda_cal=lam_cal,
        lambda_str=cfg.lambda_str,
        lambda_probe=cfg.lambda_probe,
    )
    aux.update(
        {
            "loss": loss,
            "rec": rec,
            "cal": cal,
            "str": str_loss,
            "probe": probe,
            "rgb": rgb,
            "targets": targets,
            "x_stages": x_stages,
        }
    )
    return loss, aux
