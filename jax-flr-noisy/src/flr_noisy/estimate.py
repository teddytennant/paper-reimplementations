"""Penalized trigonometric least squares and model selection (paper §2.2)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from flr_noisy.basis import (
    ALPHA,
    KAPPA,
    dimension,
    eigenvalue_threshold,
    penalty as penalty_value,
    theoretical_max_model_index,
    theoretical_recon_dim,
)
from flr_noisy.reconstruct import fourier_coefficients


@dataclass
class FitResult:
    alpha: np.ndarray
    m_hat: int
    dim: int
    recon_coeffs: np.ndarray
    recon_dim: int
    gammas: np.ndarray
    penalties: np.ndarray
    criteria: np.ndarray
    lam_mins: np.ndarray
    on_gbar: bool
    sigma2: float | None
    kappa: float
    s_n: float
    plugin: bool


def gram_and_rhs(xtilde: jnp.ndarray, y: jnp.ndarray, d_m: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Φ_m = n^{-1} ∑ x̃_i x̃_i^T and b = n^{-1} ∑ Y_i x̃_i, truncated to D_m."""
    xtilde = jnp.asarray(xtilde, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    n = y.shape[0]
    xm = xtilde[:, :d_m]
    phi = (xm.T @ xm) / n
    phi = 0.5 * (phi + phi.T)
    b = (xm.T @ y) / n
    return phi, b


def fit_fixed_m(
    xtilde: jnp.ndarray, y: jnp.ndarray, m: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (α̃, γ_n(β̂_m), λ̂_m^{(p)}) for a single trigonometric space S_m."""
    d_m = dimension(m)
    xtilde = jnp.asarray(xtilde, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    phi, b = gram_and_rhs(xtilde, y, d_m)
    lam = jnp.linalg.eigvalsh(phi)
    alpha = jnp.linalg.solve(phi, b)
    resid = y - xtilde[:, :d_m] @ alpha
    gamma = jnp.mean(resid**2)
    return alpha, gamma, lam[0]


def _zero_result(
    recon_coeffs: np.ndarray,
    recon_dim: int,
    max_m: int,
    kappa: float,
    s_n: float,
    sigma2: float | None,
    plugin: bool,
) -> FitResult:
    n_models = max_m
    empty = np.full(n_models, np.nan, dtype=np.float64)
    return FitResult(
        alpha=np.zeros(0, dtype=np.float64),
        m_hat=0,
        dim=0,
        recon_coeffs=np.asarray(recon_coeffs, dtype=np.float64),
        recon_dim=recon_dim,
        gammas=empty.copy(),
        penalties=empty.copy(),
        criteria=empty.copy(),
        lam_mins=empty.copy(),
        on_gbar=False,
        sigma2=sigma2,
        kappa=kappa,
        s_n=s_n,
        plugin=plugin,
    )


def select_model(
    xtilde: jnp.ndarray,
    y: jnp.ndarray,
    max_m: int,
    *,
    sigma2: float | None = None,
    kappa: float = KAPPA,
    alpha: float = ALPHA,
) -> FitResult:
    """Select m̂ ∈ argmin_m (γ_n(β̂_m) + pen(m)) on Ḡ, else return 0.

    If sigma2 is None, use the plug-in σ̂_m² = γ_n(β̂_m).
    """
    xtilde = jnp.asarray(xtilde, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    n = int(y.shape[0])
    recon_dim = int(xtilde.shape[1])
    max_m = min(max_m, (recon_dim - 1) // 2)
    if max_m < 1:
        raise ValueError("need recon_dim >= 3 so that S_1 is available")
    s_n = eigenvalue_threshold(n, alpha=alpha)
    plugin = sigma2 is None

    alphas: list[jnp.ndarray] = []
    gammas = np.empty(max_m, dtype=np.float64)
    lam_mins = np.empty(max_m, dtype=np.float64)
    penalties = np.empty(max_m, dtype=np.float64)
    criteria = np.empty(max_m, dtype=np.float64)

    on_gbar = True
    for m in range(1, max_m + 1):
        a_m, g_m, lam_m = fit_fixed_m(xtilde, y, m)
        alphas.append(np.asarray(a_m, dtype=np.float64))
        gammas[m - 1] = float(g_m)
        lam_mins[m - 1] = float(lam_m)
        if float(lam_m) < s_n:
            on_gbar = False
        sig_m = float(g_m) if plugin else float(sigma2)
        penalties[m - 1] = penalty_value(m, n, sig_m, kappa=kappa)
        criteria[m - 1] = gammas[m - 1] + penalties[m - 1]

    recon_np = np.asarray(xtilde, dtype=np.float64)
    if not on_gbar:
        return _zero_result(
            recon_np, recon_dim, max_m, kappa, s_n, sigma2, plugin
        )

    m_hat = int(np.argmin(criteria) + 1)
    alpha_hat = alphas[m_hat - 1]
    return FitResult(
        alpha=alpha_hat,
        m_hat=m_hat,
        dim=dimension(m_hat),
        recon_coeffs=recon_np,
        recon_dim=recon_dim,
        gammas=gammas,
        penalties=penalties,
        criteria=criteria,
        lam_mins=lam_mins,
        on_gbar=True,
        sigma2=sigma2,
        kappa=kappa,
        s_n=s_n,
        plugin=plugin,
    )


def fit_flr(
    Z: jnp.ndarray,
    y: jnp.ndarray,
    *,
    recon_dim: int | None = None,
    max_m: int | None = None,
    sigma2: float | None = None,
    kappa: float = KAPPA,
    alpha: float = ALPHA,
) -> FitResult:
    """Two-step estimator: reconstruct, then penalized trigonometric LS."""
    Z = jnp.asarray(Z, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    n, p = int(Z.shape[0]), int(Z.shape[1])
    if recon_dim is None:
        recon_dim = theoretical_recon_dim(n, p)
    if recon_dim < 1:
        raise ValueError("recon_dim must be >= 1")
    if recon_dim >= p:
        raise ValueError("need recon_dim < p for discrete orthogonality")
    xtilde = fourier_coefficients(Z, recon_dim)
    if max_m is None:
        max_m = min(
            theoretical_max_model_index(n, p),
            max((recon_dim - 1) // 2, 1),
        )
        max_m = max(max_m, 1)
    return select_model(
        xtilde, y, max_m, sigma2=sigma2, kappa=kappa, alpha=alpha
    )


def predict(alpha: jnp.ndarray, x_coeffs: jnp.ndarray) -> jnp.ndarray:
    """⟨β̂, X⟩_{L²} = ∑_{j=1}^{D} α̃_j x_j for Fourier coefficients of X."""
    alpha = jnp.asarray(alpha, dtype=jnp.float64).reshape(-1)
    x_coeffs = jnp.asarray(x_coeffs, dtype=jnp.float64)
    if alpha.size == 0:
        if x_coeffs.ndim == 1:
            return jnp.asarray(0.0, dtype=jnp.float64)
        return jnp.zeros(x_coeffs.shape[0], dtype=jnp.float64)
    d = int(alpha.shape[0])
    if x_coeffs.ndim == 1:
        if x_coeffs.shape[0] < d:
            raise ValueError("x_coeffs shorter than alpha")
        return jnp.dot(x_coeffs[:d], alpha)
    if x_coeffs.shape[1] < d:
        raise ValueError("x_coeffs shorter than alpha")
    return x_coeffs[:, :d] @ alpha


def predict_from_grid(
    alpha: jnp.ndarray, Z: jnp.ndarray, recon_dim: int | None = None
) -> jnp.ndarray:
    """Project grid observations onto the Fourier basis, then form ⟨β̂, X̃⟩."""
    alpha = jnp.asarray(alpha, dtype=jnp.float64).reshape(-1)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    if recon_dim is None:
        recon_dim = max(int(alpha.size), 1)
    recon_dim = max(recon_dim, int(alpha.size), 1)
    coeffs = fourier_coefficients(Z, recon_dim)
    return predict(alpha, coeffs)
