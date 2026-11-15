"""Ridge weak-form estimator, gradient-only baseline, and Prop. 15 audit."""
from __future__ import annotations

from . import _host

_host.preload()

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def antisym(H: jax.Array) -> jax.Array:
    """Solenoidal (antisymmetric) part of a linear map."""
    return 0.5 * (H - H.T)


def sym(H: jax.Array) -> jax.Array:
    return 0.5 * (H + H.T)


def vec(H: jax.Array) -> jax.Array:
    """Row-major vec, matching moments.weak_design_from_second_moments."""
    return H.reshape(-1)


def unvec(a: jax.Array, d: int) -> jax.Array:
    return a.reshape(d, d)


def colvec(H: jax.Array) -> jax.Array:
    """Column-stacked vec, matching Prop. 15."""
    return H.T.reshape(-1)


def ridge_solve(X: jax.Array, bb: jax.Array, zeta: float | jax.Array = 0.0) -> jax.Array:
    """Unweighted ridge: argmin ||X a - bb||^2 + zeta ||a||^2."""
    zeta = jnp.asarray(zeta, dtype=X.dtype)
    gram = X.T @ X
    p = gram.shape[0]
    return jnp.linalg.solve(gram + zeta * jnp.eye(p, dtype=X.dtype), X.T @ bb)


def _basis_from_matrices(mats: list[jax.Array], col: bool = False) -> jax.Array:
    flatten = colvec if col else vec
    cols = [flatten(G) for G in mats]
    P = jnp.stack(cols, axis=1)
    norms = jnp.linalg.norm(P, axis=0, keepdims=True)
    return P / jnp.clip(norms, 1e-18, None)


def antisymmetric_basis(d: int, col: bool = False) -> jax.Array:
    mats = []
    eye_like = jnp.zeros((d, d), dtype=jnp.float64)
    for i in range(d):
        for j in range(i + 1, d):
            G = eye_like.at[i, j].set(1.0).at[j, i].set(-1.0)
            mats.append(G)
    return _basis_from_matrices(mats, col=col)


def symmetric_basis(d: int, col: bool = False) -> jax.Array:
    mats = []
    eye_like = jnp.zeros((d, d), dtype=jnp.float64)
    for i in range(d):
        mats.append(eye_like.at[i, i].set(1.0))
        for j in range(i + 1, d):
            G = eye_like.at[i, j].set(1.0).at[j, i].set(1.0)
            mats.append(G)
    return _basis_from_matrices(mats, col=col)


def ridge_fit(
    X: jax.Array,
    bb: jax.Array,
    d: int,
    zeta: float | jax.Array = 0.0,
    restrict: str = "full",
) -> jax.Array:
    """Fit F(x) = H x. restrict in {full, symmetric, antisymmetric}."""
    if restrict == "full":
        a = ridge_solve(X, bb, zeta=zeta)
        return unvec(a, d)
    if restrict == "symmetric":
        P = symmetric_basis(d, col=False)
    elif restrict == "antisymmetric":
        P = antisymmetric_basis(d, col=False)
    else:
        raise ValueError(f"unknown restrict={restrict}")
    alpha = ridge_solve(X @ P, bb, zeta=zeta)
    return unvec(P @ alpha, d)


def rel_err_solenoidal(H_hat: jax.Array, H_star: jax.Array) -> jax.Array:
    """||ĥ - h*||_F / ||h*||_F on the antisymmetric part."""
    h_hat = antisym(H_hat)
    h_star = antisym(H_star)
    denom = jnp.linalg.norm(h_star)
    return jnp.linalg.norm(h_hat - h_star) / jnp.clip(denom, 1e-18, None)


def gradient_only_fit(
    X: jax.Array,
    bb: jax.Array,
    d: int,
    zeta: float | jax.Array = 0.0,
) -> jax.Array:
    """Gradient-only linear field: H constrained to be symmetric (curl-blind)."""
    return ridge_fit(X, bb, d, zeta=zeta, restrict="symmetric")


def spd_sqrt(S: jax.Array) -> jax.Array:
    w, V = jnp.linalg.eigh(S)
    w = jnp.clip(w, 1e-18, None)
    return (V * jnp.sqrt(w)) @ V.T


def gaussian_ot_map(S0: jax.Array, S1: jax.Array) -> jax.Array:
    """Centered-Gaussian OT map T(x) = A x, A SPD."""
    s0h = spd_sqrt(S0)
    s0ih = spd_sqrt(jnp.linalg.inv(S0))
    inner = s0h @ S1 @ s0h
    return s0ih @ spd_sqrt(inner) @ s0ih


def displacement_velocity(S0: jax.Array, S1: jax.Array, dt: jax.Array) -> jax.Array:
    """Linear velocity of the displacement interpolant, (A - I)/Δt."""
    A = gaussian_ot_map(S0, S1)
    d = S0.shape[0]
    return (A - jnp.eye(d, dtype=S0.dtype)) / dt


def mean_displacement_field(Sigmas: jax.Array, dts: jax.Array) -> jax.Array:
    """Average displacement velocity across consecutive Gaussian slices."""

    def one(S0, S1, dt):
        return displacement_velocity(S0, S1, dt)

    Vs = jax.vmap(one)(Sigmas[:-1], Sigmas[1:], dts)
    return jnp.mean(Vs, axis=0)


def commutation_matrix(d: int) -> jax.Array:
    """C vec_col(G) = vec_col(G^T)."""
    C = jnp.zeros((d * d, d * d), dtype=jnp.float64)
    for i in range(d):
        for j in range(d):
            row = j + d * i
            col = i + d * j
            C = C.at[row, col].set(1.0)
    return C


def prop15_hessian(Sigmas: jax.Array) -> jax.Array:
    """Strong-form unweighted Hessian of Prop. 15, column-stacked vec.

    (1/K) sum_k (Sigma_k ⊗ Sigma_k^{-1} + C_m).
    """
    K, d, _ = Sigmas.shape
    C = commutation_matrix(d)

    def term(S):
        Sinv = jnp.linalg.inv(S)
        return jnp.kron(S, Sinv) + C

    return jnp.mean(jax.vmap(term)(Sigmas), axis=0)


def prop15_schur_eigs(Sigmas: jax.Array) -> jax.Array:
    """Eigenvalues of the profiled solenoidal curvature (Schur complement)."""
    d = Sigmas.shape[-1]
    Hess = prop15_hessian(Sigmas)
    P_anti = antisymmetric_basis(d, col=True)
    P_sym = symmetric_basis(d, col=True)
    P = jnp.concatenate([P_anti, P_sym], axis=1)
    Q = P.T @ Hess @ P
    n_a = P_anti.shape[1]
    Haa = Q[:n_a, :n_a]
    Has = Q[:n_a, n_a:]
    Hss = Q[n_a:, n_a:]
    schur = Haa - Has @ jnp.linalg.pinv(Hss, rcond=1e-12) @ Has.T
    eigs = jnp.linalg.eigvalsh(0.5 * (schur + schur.T))
    return jnp.sort(eigs)


def weak_solenoidal_gram_eigs(X: jax.Array, d: int) -> jax.Array:
    """Eigenvalues of X^T X on the antisymmetric block (weak-form design)."""
    P = antisymmetric_basis(d, col=False)
    G = P.T @ (X.T @ X) @ P
    eigs = jnp.linalg.eigvalsh(0.5 * (G + G.T))
    return jnp.sort(eigs)
