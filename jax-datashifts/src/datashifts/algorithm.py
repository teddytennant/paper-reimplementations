"""DataShifts Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from .estimators import Array, CostFn, concept_shift, debiased_w_beta
from .sinkhorn import entropic_ot


@dataclass
class DataShiftsResult:
    s_cov: Array
    s_cpt: Array
    bound: Optional[Array] = None
    w_plugin: Optional[Array] = None
    gamma: Optional[Array] = None


def datashifts(
    x_s: Array,
    y_s: Array,
    x_t: Array,
    y_t: Array,
    beta: float = 0.01,
    rho_x: Optional[CostFn] = None,
    rho_y: Optional[CostFn] = None,
    n_iter: int = 2000,
    lipschitz_ell: Optional[float] = None,
    lipschitz_ell_prime: Optional[float] = None,
    lipschitz_h: Optional[float] = None,
    source_error: Optional[float] = None,
    return_plugin: bool = False,
) -> DataShiftsResult:
    """Algorithm 1.

    Ŝ_Cov = W_β^deb (Def. 4.1); Ŝ_Cpt from Def. 4.3.
    If Lipschitz constants and ε̂_S are given:
        B = ε̂_S + L_h L_ℓ' Ŝ_Cov + L_ℓ Ŝ_Cpt
    """
    s_cov = debiased_w_beta(x_s, x_t, beta=beta, rho=rho_x, n_iter=n_iter)
    w_plugin, gamma = entropic_ot(x_s, x_t, beta=beta, rho=rho_x, n_iter=n_iter)
    s_cpt = concept_shift(
        x_s,
        y_s,
        x_t,
        y_t,
        beta=beta,
        rho_x=rho_x,
        rho_y=rho_y,
        n_iter=n_iter,
        gamma=gamma,
    )
    bound = None
    given = (lipschitz_ell, lipschitz_ell_prime, lipschitz_h, source_error)
    if all(v is not None for v in given):
        bound = (
            jnp.asarray(source_error)
            + jnp.asarray(lipschitz_h) * jnp.asarray(lipschitz_ell_prime) * s_cov
            + jnp.asarray(lipschitz_ell) * s_cpt
        )
    return DataShiftsResult(
        s_cov=s_cov,
        s_cpt=s_cpt,
        bound=bound,
        w_plugin=w_plugin if return_plugin else None,
        gamma=gamma if return_plugin else None,
    )
