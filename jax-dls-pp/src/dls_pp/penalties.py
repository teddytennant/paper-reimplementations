"""Roughness and Gaussian-process penalties (eq 2.7–2.11)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from dls_pp.splines import SPLINE_DEGREE, basis_derivative, basis_second_derivative, n_basis


def gauss_legendre(n: int, a: float, b: float) -> tuple[Array, Array]:
    """n-point Gauss–Legendre nodes and weights on [a, b]."""
    n = int(n)
    k = jnp.arange(1, n, dtype=jnp.float64)
    beta = k / jnp.sqrt(4.0 * k * k - 1.0)
    jac = jnp.diag(beta, 1) + jnp.diag(beta, -1)
    nodes, vecs = jnp.linalg.eigh(jac)
    weights = 2.0 * vecs[0, :] ** 2
    scale = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return nodes * scale + mid, weights * scale


def omega_matrices(
    knots: Array,
    t_max: float,
    n_quad: int = 64,
    degree: int = SPLINE_DEGREE,
) -> tuple[Array, Array]:
    """Ω1 = ∫ φ' φ'^T dt and Ω2 = ∫ φ'' φ''^T dt on [0, T] (eq 2.7–2.8)."""
    times, weights = gauss_legendre(n_quad, 0.0, float(t_max))
    d1 = jnp.stack([basis_derivative(t, knots, degree) for t in times])
    d2 = jnp.stack([basis_second_derivative(t, knots, degree) for t in times])
    omega1 = d1.T @ (weights[:, None] * d1)
    omega2 = d2.T @ (weights[:, None] * d2)
    return omega1, omega2


def f1(c_i_x: Array, c_i_y: Array, omega1: Array) -> Array:
    """∫ (ẋ_i² + ẏ_i²) dt = c_i^x^T Ω1 c_i^x + c_i^y^T Ω1 c_i^y (eq 2.7)."""
    return c_i_x @ omega1 @ c_i_x + c_i_y @ omega1 @ c_i_y


def f2(c_i_x: Array, c_i_y: Array, omega2: Array) -> Array:
    """∫ (ẍ_i² + ÿ_i²) dt = c_i^x^T Ω2 c_i^x + c_i^y^T Ω2 c_i^y (eq 2.8)."""
    return c_i_x @ omega2 @ c_i_x + c_i_y @ omega2 @ c_i_y


def penalty_matrix(omega1: Array, omega2: Array, tau: float, rho1: float, rho2: float) -> Array:
    """Π = τ^{-2} I_K + 2 ρ1 Ω1 + 2 ρ2 Ω2 (eq 2.10)."""
    k = omega1.shape[0]
    return (1.0 / (tau * tau)) * jnp.eye(k, dtype=omega1.dtype) + 2.0 * rho1 * omega1 + 2.0 * rho2 * omega2


def roughness_penalty(
    c_x: Array,
    c_y: Array,
    omega1: Array,
    omega2: Array,
    tau: float,
    rho1: float,
    rho2: float,
) -> Array:
    """P(c) = (1/(2τ²)) ||c||_F² + ρ1 ∑_i f1(c_i) + ρ2 ∑_i f2(c_i) (eq 2.9)."""
    fro = jnp.sum(c_x * c_x) + jnp.sum(c_y * c_y)
    speed = jnp.sum(jnp.einsum("ik,kl,il->i", c_x, omega1, c_x) + jnp.einsum("ik,kl,il->i", c_y, omega1, c_y))
    acc = jnp.sum(jnp.einsum("ik,kl,il->i", c_x, omega2, c_x) + jnp.einsum("ik,kl,il->i", c_y, omega2, c_y))
    return (0.5 / (tau * tau)) * fro + rho1 * speed + rho2 * acc


def quadratic_form_penalty(c_x: Array, c_y: Array, pi: Array) -> Array:
    """Equivalent form (1/2) ∑_i (c_i^x^T Π c_i^x + c_i^y^T Π c_i^y)."""
    qx = jnp.einsum("ik,kl,il->i", c_x, pi, c_x)
    qy = jnp.einsum("ik,kl,il->i", c_y, pi, c_y)
    return 0.5 * jnp.sum(qx + qy)


def is_positive_definite(pi: Array, tol: float = 1e-10) -> Array:
    """True iff Π ≻ 0."""
    eig = jnp.linalg.eigvalsh(pi)
    return jnp.all(eig > tol)


def _unused_n_basis() -> int:
    return n_basis
