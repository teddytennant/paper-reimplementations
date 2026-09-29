"""Benjamini-Yekutieli / Benjamini-Hochberg step-up and Algorithm 1 (DPS)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

import jax.numpy as jnp
from jax import Array

from gp_dps.bootstrap import loo_residuals, multiplier_bootstrap, studentized_pvalues
from gp_dps.gp import fit_se_ard
from gp_dps.sensitivity import dps_maps


def _step_up(p: np.ndarray, q: float, harmonic: bool) -> np.ndarray:
    """BH (harmonic=False) or BY (harmonic=True) step-up. Returns reject mask."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    p_sorted = p[order]
    c_m = float(np.sum(1.0 / np.arange(1, m + 1))) if harmonic else 1.0
    thresh = (np.arange(1, m + 1) / m) * q / c_m
    below = p_sorted <= thresh
    selected = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.where(below)[0].max()) + 1
        selected[order[:k]] = True
    return selected


def bh_reject(p: ArrayLike, q: float = 0.2) -> np.ndarray:
    """Benjamini-Hochberg step-up at level q."""
    return _step_up(np.asarray(p), q, harmonic=False)


def by_reject(p: ArrayLike, q: float = 0.2) -> np.ndarray:
    """Benjamini-Yekutieli step-up at level q (Algorithm 1 default)."""
    return _step_up(np.asarray(p), q, harmonic=True)


def selected_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def dps_from_params(
    X: Array,
    y: Array,
    ell: Array,
    eta: Array,
    sigma2: Array,
    key: Array,
    B: int = 32,
    q: float = 0.2,
    procedure: str = "by",
) -> dict:
    """Algorithm 1 steps 2--7 given fitted hyperparameters."""
    maps = dps_maps(X, y, ell, eta, sigma2)
    r = loo_residuals(maps["L"], y)
    nu_boot = multiplier_bootstrap(maps["A"], r, key, B)
    p, T, T_boot, bbar, s = studentized_pvalues(maps["nu"], nu_boot)
    p_np = np.asarray(p)
    if procedure == "bh":
        mask = bh_reject(p_np, q)
    else:
        mask = by_reject(p_np, q)
    return {
        "nu": maps["nu"],
        "nu_boot": nu_boot,
        "p": p,
        "T": T,
        "T_boot": T_boot,
        "bbar": bbar,
        "s": s,
        "r": r,
        "A": maps["A"],
        "gbar": maps["gbar"],
        "v": maps["v"],
        "plugin": maps["plugin"],
        "ell": ell,
        "eta": eta,
        "sigma2": sigma2,
        "mask": mask,
        "selected": selected_indices(mask),
    }


def dps(
    X: Array,
    y: Array,
    key: Array,
    B: int = 32,
    q: float = 0.2,
    procedure: str = "by",
    steps: int = 80,
    lr: float = 0.08,
) -> dict:
    """Algorithm 1: fit SE-ARD, DPS maps, LOO multiplier bootstrap, BY/BH."""
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    ell, eta, sigma2 = fit_se_ard(X, y, steps=steps, lr=lr)
    return dps_from_params(X, y, ell, eta, sigma2, key, B=B, q=q, procedure=procedure)
