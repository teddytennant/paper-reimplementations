"""Block-averaged DPS (paper Appendix I)."""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from gp_dps.bootstrap import studentized_pvalues
from gp_dps.gp import fit_se_ard
from gp_dps.select import bh_reject, by_reject, dps_from_params, selected_indices


def average_block_sensitivities(
    blocks: list[tuple[Array, Array]],
    ell: Array,
    eta: Array,
    sigma2: Array,
    key: Array,
    B: int = 32,
    q: float = 0.2,
    procedure: str = "by",
) -> dict:
    """Exact DPS in each block with independent multipliers; average then studentize."""
    nu_list = []
    boot_list = []
    keys = jax.random.split(key, len(blocks))
    for (Xb, yb), kb in zip(blocks, keys):
        out = dps_from_params(Xb, yb, ell, eta, sigma2, kb, B=B, q=q, procedure=procedure)
        nu_list.append(out["nu"])
        boot_list.append(out["nu_boot"])
    nu = jnp.mean(jnp.stack(nu_list, axis=0), axis=0)
    nu_boot = jnp.mean(jnp.stack(boot_list, axis=0), axis=0)
    p, T, T_boot, bbar, s = studentized_pvalues(nu, nu_boot)
    p_np = np.asarray(p)
    mask = bh_reject(p_np, q) if procedure == "bh" else by_reject(p_np, q)
    return {
        "nu": nu,
        "nu_boot": nu_boot,
        "nu_blocks": jnp.stack(nu_list, axis=0),
        "p": p,
        "T": T,
        "T_boot": T_boot,
        "bbar": bbar,
        "s": s,
        "ell": ell,
        "eta": eta,
        "sigma2": sigma2,
        "mask": mask,
        "selected": selected_indices(mask),
    }


def split_blocks(X: Array, y: Array, n_blocks: int, holdout: int, key: Array):
    """Shuffle, hold out `holdout` points for hyperparameter fit, split the rest."""
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    X = X[perm]
    y = y[perm]
    Xh, yh = X[:holdout], y[:holdout]
    Xr, yr = X[holdout:], y[holdout:]
    rest = Xr.shape[0]
    sizes = [rest // n_blocks] * n_blocks
    for i in range(rest % n_blocks):
        sizes[i] += 1
    blocks = []
    start = 0
    for sz in sizes:
        blocks.append((Xr[start : start + sz], yr[start : start + sz]))
        start += sz
    return (Xh, yh), blocks


def block_dps(
    X: Array,
    y: Array,
    key: Array,
    n_blocks: int = 2,
    holdout_frac: float = 0.3,
    B: int = 32,
    q: float = 0.2,
    procedure: str = "by",
    steps: int = 80,
    lr: float = 0.08,
) -> dict:
    """Appendix I: fit on a held-out subset, exact DPS per remaining block, average."""
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n = X.shape[0]
    holdout = max(int(holdout_frac * n), X.shape[1] + 2)
    holdout = min(holdout, n - n_blocks)
    k_fit, k_boot = jax.random.split(key)
    (Xh, yh), blocks = split_blocks(X, y, n_blocks, holdout, k_fit)
    ell, eta, sigma2 = fit_se_ard(Xh, yh, steps=steps, lr=lr)
    out = average_block_sensitivities(blocks, ell, eta, sigma2, k_boot, B=B, q=q, procedure=procedure)
    out["holdout"] = holdout
    out["n_blocks"] = n_blocks
    return out
