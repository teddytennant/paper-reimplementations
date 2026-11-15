"""Tiny offline + online ADE run. Prints rRMSE against Ogata–Banks."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from jax_lpinn.data import (
    bc_dirichlet_collocation,
    bc_neumann_collocation,
    flatten_mesh,
    ic_collocation,
    interior_collocation,
    make_ade_ensemble,
    split_mean_fluct,
)
from jax_lpinn.metrics import rrmse
from jax_lpinn.model import LPINNConfig, init_fluct_hidden, init_mean_params, predict
from jax_lpinn.physics import ogata_banks
from jax_lpinn.train import online_forward_lstsq, train_mean_derivative_matching


def main():
    cfg = LPINNConfig(hidden_widths=(8, 8), lam_t=0.0, lam_x=0.0, lam_xx=0.0)
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    mean = init_mean_params(k1, cfg)
    hidden = init_fluct_hidden(k2, cfg)

    Nx, Nt, n_train = 16, 8, 8
    ens = make_ade_ensemble(k3, n_train, Nx, Nt, cfg, with_derivs=False)
    split = split_mean_fluct(ens["U"])
    x_flat, t_flat = flatten_mesh(ens["xx"], ens["tt"])
    u_bar = split["U_bar"].reshape(-1)
    z = jnp.zeros_like(u_bar)
    mean, losses = train_mean_derivative_matching(
        mean, x_flat, t_flat, u_bar, z, z, z, cfg, steps=8, lr=5e-3
    )

    x_res, t_res = interior_collocation(Nx, Nt, cfg.L, cfg.T)
    x_ic, t_ic = ic_collocation(Nx, cfg.L)
    x_bc0, t_bc0 = bc_dirichlet_collocation(Nt, cfg.T)
    x_bcL, t_bcL = bc_neumann_collocation(Nt, cfg.L, cfg.T)

    V, D = 1.05 * cfg.V_star, 0.95 * cfg.D_star
    W = online_forward_lstsq(
        mean, hidden, x_res, t_res, x_ic, t_ic, x_bc0, t_bc0, x_bcL, t_bcL, V, D, cfg
    )
    u_pred = predict(mean, hidden, W, x_flat, t_flat, L=cfg.L, T=cfg.T)
    u_ref = ogata_banks(x_flat, t_flat, V, D, cfg.u0)
    err = float(rrmse(u_pred, u_ref, cfg.u0))
    print(f"offline_loss_last={float(losses[-1]):.4e}")
    print(f"rRMSE={err:.4e}")
    print(f"Neta={W.shape[0]} Ntrain={n_train} Nx={Nx} Nt={Nt}")


if __name__ == "__main__":
    main()
