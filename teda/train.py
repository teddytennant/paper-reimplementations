"""CPU smoke: synthesize a few pixels, run TEDA, write results.json."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from teda.match import run_batch
from teda.operators import build_stage1_operators
from teda.planck import linear_in_T
from teda.synth import (
    channel_grid,
    channel_kappa,
    fine_grid,
    observe_radiance,
    sensor_transmittance,
    synthetic_kappa,
    transmittance_bank,
)


def main():
    lam_ch = channel_grid(48)
    lam_fine = fine_grid(801)
    kappa_fine = synthetic_kappa(lam_fine)
    alpha_tilde = channel_kappa(lam_fine, kappa_fine, lam_ch, fwhm_um=None)
    ops = build_stage1_operators(alpha_tilde)

    d_grid = jnp.array([50.0, 100.0, 150.0, 200.0])
    tau_bank = transmittance_bank(d_grid, lam_fine, kappa_fine, lam_ch, fwhm_um=None)

    T_air = 290.0
    B_air = linear_in_T(T_air, scale=1.0, like=lam_ch)
    true_d = jnp.array([50.0, 100.0, 150.0, 200.0, 100.0, 150.0])
    temps = jnp.array([305.0, 300.0, 310.0, 298.0, 308.0, 302.0])

    def _synth(d, T):
        tau = sensor_transmittance(d, lam_fine, kappa_fine, lam_ch, fwhm_um=None)
        B_obj = linear_in_T(T, scale=1.0, like=lam_ch)
        return observe_radiance(tau, B_obj, B_air, emissivity=1.0)

    y_batch = jax.vmap(_synth)(true_d, temps)
    out = run_batch(y_batch, B_air, ops, tau_bank, d_grid, 0.0)

    true_np = np.asarray(true_d)
    hat_np = np.asarray(out.d_hat)
    rmse = float(np.sqrt(np.mean((hat_np - true_np) ** 2)))
    payload = {
        "true_d": true_np.tolist(),
        "hat_d": hat_np.tolist(),
        "rmse": rmse,
    }
    path = Path(__file__).resolve().parent / "results.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    print(f"true_d={payload['true_d']}")
    print(f"hat_d={payload['hat_d']}")
    print(f"rmse={rmse:.4f}")


if __name__ == "__main__":
    main()
