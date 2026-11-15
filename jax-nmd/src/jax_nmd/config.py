"""Tiny Neural Modal Decomposition hyperparameters."""

from __future__ import annotations

from typing import NamedTuple


class NMDConfig(NamedTuple):
    geom_h: int = 8
    geom_w: int = 8
    d_t: int = 32
    n_tokens: int = 4
    k_poles: int = 4
    hidden: int = 64
    conv_mid: int = 16
    port_dim: int = 2
    z0: float = 50.0
    lambda_z: float = 0.1
    lambda_g: float = 0.5
    eps_z: float = 1e-2
    eps_anchor: float = 1e-6
    lr: float = 3e-4
    weight_decay: float = 1e-3
    ln_eps: float = 1e-5
    # Paper default for 18-row cavities. This kernel uses `is_anchor` instead.
    r_anchor: int = 17
