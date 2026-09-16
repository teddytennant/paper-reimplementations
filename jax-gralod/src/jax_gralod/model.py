"""GraLoD config, parameter init, training and inference forwards.

Backbone-agnostic: `forward_infer` / `forward_train_features` take lists of
encoder features F^l and decoder features D_s. A tiny dummy restorer that
wires a conv encoder/decoder around these hooks lives in `backbone.py`.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from jax_gralod.align import align_pyramid, init_aligner, map_pyramid_to_stage
from jax_gralod.estimator import apply_probe, init_adaptor, init_estimator, init_probe, inject_stage
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
from jax_gralod.ops import Array, Params


class GraLoDConfig(NamedTuple):
    encoder_channels: tuple[int, ...] = (8, 16, 32)
    decoder_channels: tuple[int, ...] = (8, 16, 32)
    lod_channels: int = 8
    in_channels: int = 3
    out_channels: int = 3
    gamma: float = 0.05
    tau: float = 0.20
    beta: float = 10.0
    lambda_cal: float = 0.10
    lambda_str: float = 0.05
    lambda_probe: float = 0.10
    box_ksize: int = 5
    ln_eps: float = 1e-5
    warmup_steps: int = 0

    @property
    def n_levels(self) -> int:
        return len(self.encoder_channels)

    @property
    def n_stages(self) -> int:
        return len(self.decoder_channels)


def init(key: Array, cfg: GraLoDConfig) -> Params:
    """Init aligners A_l and per-stage (G_s, H_s, P_s)."""
    n_levels = cfg.n_levels
    n_stages = cfg.n_stages
    k_align, k_stages = jax.random.split(key)
    align_keys = jax.random.split(k_align, n_levels)
    align = [
        init_aligner(align_keys[l], cfg.encoder_channels[l], cfg.lod_channels)
        for l in range(n_levels)
    ]
    stage_keys = jax.random.split(k_stages, n_stages)
    stages: list[Params] = []
    for s in range(n_stages):
        kg, kh, kp = jax.random.split(stage_keys[s], 3)
        dec_ch = cfg.decoder_channels[s]
        stages.append(
            {
                "G": init_estimator(kg, dec_ch, cfg.lod_channels, n_levels),
                "H": init_adaptor(kh, dec_ch, cfg.lod_channels),
                "P": init_probe(kp, dec_ch, cfg.lod_channels, cfg.out_channels),
            }
        )
    return {"align": align, "stages": stages}


def drop_probes(params: Params) -> Params:
    """Inference pytree: drop P_s. `forward_infer` never reads probe weights."""
    stages = [{k: v for k, v in st.items() if k != "P"} for st in params["stages"]]
    return {"align": params["align"], "stages": stages}


def _inject_all(
    params: Params,
    encoder_features: Sequence[Array],
    decoder_features: Sequence[Array],
    cfg: GraLoDConfig,
) -> tuple[list[Array], list[Array], list[tuple[Array, ...]], tuple[Array, ...]]:
    z = align_pyramid(params["align"], encoder_features, ln_eps=cfg.ln_eps)
    d_tildes: list[Array] = []
    lambdas: list[Array] = []
    z_stages: list[tuple[Array, ...]] = []
    for s, d_s in enumerate(decoder_features):
        _, h, w, _ = d_s.shape
        z_s = map_pyramid_to_stage(z, h, w)
        d_t, lam, _f = inject_stage(params["stages"][s], d_s, z_s, cfg.n_levels)
        d_tildes.append(d_t)
        lambdas.append(lam)
        z_stages.append(z_s)
    return d_tildes, lambdas, z_stages, z


def forward_infer(
    params: Params,
    encoder_features: Sequence[Array],
    decoder_features: Sequence[Array],
    cfg: GraLoDConfig,
) -> tuple[list[Array], list[Array]]:
    """Inference: A_l, R_{s,l}, G_s, query, H_s. No probes, no MSFC, no SAR."""
    d_tildes, lambdas, _z_s, _z = _inject_all(
        params, encoder_features, decoder_features, cfg
    )
    return d_tildes, lambdas


def _stage_probes(
    stage_params: Params,
    decoder_feat: Array,
    z_stage: Sequence[Array],
    detach_inputs: bool = True,
) -> tuple[Array, ...]:
    return tuple(
        apply_probe(stage_params["P"], decoder_feat, z_l, detach_inputs=detach_inputs)
        for z_l in z_stage
    )


def forward_train(
    params: Params,
    encoder_features: Sequence[Array],
    decoder_features: Sequence[Array],
    restored: Array,
    clean: Array,
    cfg: GraLoDConfig,
    step: int = 0,
) -> tuple[Array, dict[str, Any]]:
    """Training forward given backbone features and a restored image.

    Probes use stop-gradient on Concat(D_s, Z_s^l) for both L_probe and the
    MSFC errors so the auxiliary heads cannot hijack the backbone.
    """
    d_tildes, lambdas, z_stages, _z = _inject_all(
        params, encoder_features, decoder_features, cfg
    )
    x_stages = []
    probe_imgs: list[tuple[Array, ...]] = []
    targets = []
    for s, d_s in enumerate(decoder_features):
        _, h, w, _ = d_s.shape
        x_s = resize_clean(clean, h, w)
        x_stages.append(x_s)
        xhats = _stage_probes(params["stages"][s], d_s, z_stages[s], detach_inputs=True)
        probe_imgs.append(xhats)
        errors = stack_level_errors(xhats, x_s, ksize=cfg.box_ksize)
        lam_star, _q, _c = msfc_target(errors, gamma=cfg.gamma, tau=cfg.tau)
        targets.append(lam_star)

    rec = reconstruction_loss(restored, clean)
    cal = l_cal(lambdas, targets)
    str_loss = l_str(lambdas, x_stages, beta=cfg.beta)
    probe = l_probe(probe_imgs, x_stages)
    lam_cal = calibration_ramp(step, cfg.warmup_steps, cfg.lambda_cal)
    # When warmup_steps==0, ramp is 1 and lam_cal equals cfg.lambda_cal.
    loss = total_loss(
        rec,
        cal,
        str_loss,
        probe,
        lambda_cal=lam_cal,
        lambda_str=cfg.lambda_str,
        lambda_probe=cfg.lambda_probe,
    )
    aux = {
        "loss": loss,
        "rec": rec,
        "cal": cal,
        "str": str_loss,
        "probe": probe,
        "lambda_cal": lam_cal,
        "lambdas": lambdas,
        "d_tildes": d_tildes,
        "targets": targets,
        "x_stages": x_stages,
        "probes": probe_imgs,
    }
    return loss, aux
