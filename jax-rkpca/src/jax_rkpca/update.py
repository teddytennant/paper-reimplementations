"""Algorithm 1: rotation-based online robust kernel PCA.

Gradient of the robust objective is not fully written in the paper.
From z = (1/2) ||(I - P) x||^2 with x = h - mu and P = Gamma Gamma^T,
equivalently z = (1/2)(||x||^2 - ||Gamma^T x||^2) when Gamma has
orthonormal columns:

    dz/dmu    = -(I - P) x
    dz/dGamma = -x (Gamma^T x)^T

The robust-weighted gradients are g = Psi_dot(z) times those derivatives.
The Euclidean step is then mu <- mu - eta g_mu, Gamma <- Gamma - eta g_Gamma,
followed by re-orthonormalization of Gamma (paper line 11), then Type I
and Type II rotations with alpha_t = alpha0 * Psi_dot(z).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from jax_rkpca.distance import half_squared_distance, loadings
from jax_rkpca.kernel import empirical_kernel_map
from jax_rkpca.robust import DEFAULT_C, psi_dot
from jax_rkpca.rotation import apply_plane_rotation, type_i_vectors, type_ii_vector


class RKPCState(NamedTuple):
    """Running mean and orthonormal subspace basis in feature space."""

    mu: Array
    Gamma: Array


def orthonormalize(Gamma: Array) -> Array:
    """Thin QR so columns of Gamma are orthonormal."""
    Gamma = jnp.asarray(Gamma)
    Q, _ = jnp.linalg.qr(Gamma, mode="reduced")
    return Q


def init_state(M: int, d: int, Gamma0: Array | None = None) -> RKPCState:
    """Initialize mu = 0 and Gamma as the first d standard basis columns.

    If Gamma0 is given it is re-orthonormalized. Paper Algorithm 1 line 1.
    """
    if d > M:
        raise ValueError(f"need M >= d, got M={M}, d={d}")
    mu = jnp.zeros((M,))
    if Gamma0 is None:
        Gamma = jnp.eye(M, d)
    else:
        Gamma = orthonormalize(jnp.asarray(Gamma0))
    return RKPCState(mu=mu, Gamma=Gamma)


def subspace_gradients(
    h: Array, mu: Array, Gamma: Array
) -> tuple[Array, Array]:
    """Euclidean gradients of z with respect to mu and Gamma.

    dz/dmu = -(I - P) x, dz/dGamma = -x (Gamma^T x)^T.
    """
    h = jnp.asarray(h)
    mu = jnp.asarray(mu)
    Gamma = jnp.asarray(Gamma)
    x = h - mu
    loadings_x = Gamma.T @ x
    residual = x - Gamma @ loadings_x
    dmu = -residual
    dGamma = -jnp.outer(x, loadings_x)
    return dmu, dGamma


def gradient_step(
    mu: Array,
    Gamma: Array,
    h: Array,
    eta: float,
    weight: Array | float,
) -> tuple[Array, Array]:
    """One robust-weighted gradient step, before re-orthonormalization."""
    dmu, dGamma = subspace_gradients(h, mu, Gamma)
    wgt = jnp.asarray(weight)
    mu_new = mu - eta * wgt * dmu
    Gamma_new = Gamma - eta * wgt * dGamma
    return mu_new, Gamma_new


def step(
    state: RKPCState,
    h: Array,
    eta: float,
    alpha0: float,
    c: float = DEFAULT_C,
    rotate: bool = True,
    type_i: bool = True,
    type_ii: bool = True,
) -> tuple[RKPCState, Array, dict]:
    """One sample of Algorithm 1.

    Args:
        state: current (mu, Gamma).
        h: empirical kernel map of the new sample, shape (M,).
        eta: gradient learning rate (held fixed, paper Section IV-G).
        alpha0: base rotation factor.
        c: robust-loss scale in Psi_dot(z) = 1/(c+z).
        rotate: if False, skip Type I and Type II (gradient-only baseline).
        type_i, type_ii: which rotations to apply when ``rotate`` is True.

    Returns:
        (new_state, c_t, info) where c_t are the loadings of shape (d,)
        and info holds z, the influence weight, and alpha_t.
    """
    h = jnp.asarray(h)
    mu, Gamma = state.mu, state.Gamma
    z = half_squared_distance(h, mu, Gamma)
    weight = psi_dot(z, c=c)
    mu, Gamma = gradient_step(mu, Gamma, h, eta, weight)
    Gamma = orthonormalize(Gamma)

    alpha_t = alpha0 * weight
    if rotate:
        w = h - mu
        if type_i:
            v, _ = type_i_vectors(h, mu, Gamma)
            Gamma = apply_plane_rotation(Gamma, v, w, alpha_t)
        if type_ii:
            v = type_ii_vector(w, Gamma)
            Gamma = apply_plane_rotation(Gamma, v, w, alpha_t)
        # Paper says a rotation preserves orthonormality. Re-orth here is
        # only for accumulated floating-point drift.
        Gamma = orthonormalize(Gamma)

    new_state = RKPCState(mu=mu, Gamma=Gamma)
    c_t = loadings(h, mu, Gamma)
    info = {"z": z, "weight": weight, "alpha": alpha_t}
    return new_state, c_t, info


def track(
    X: Array,
    U: Array,
    d: int,
    eta: float,
    alpha0: float,
    kernel: str = "linear",
    sigma: float = 1.0,
    c: float = DEFAULT_C,
    rotate: bool = True,
    Gamma0: Array | None = None,
) -> tuple[RKPCState, Array]:
    """Run Algorithm 1 over a stream of input samples X, shape (n, D).

    Returns the final state and loadings of shape (n, d).
    """
    X = jnp.asarray(X)
    U = jnp.asarray(U)
    n = X.shape[0]
    M = U.shape[0]
    state = init_state(M, d, Gamma0=Gamma0)
    cs = []
    for t in range(n):
        h = empirical_kernel_map(X[t], U, kernel=kernel, sigma=sigma)
        state, c_t, _ = step(
            state, h, eta=eta, alpha0=alpha0, c=c, rotate=rotate
        )
        cs.append(c_t)
    return state, jnp.stack(cs, axis=0)
