"""Rotation-matrix construction and Type I / Type II updates, paper eqs. (7)-(11).

Section IV-F: never form the dense M x M matrix A on the hot path. The
plane rotation is applied directly to Gamma (M x d) in O(M d).

A dense A is still available for tiny tests that check A @ Gamma against
the optimized apply.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

_EPS = 1e-12


def angle_between(v: Array, w: Array, eps: float = _EPS) -> Array:
    """Angle theta between v and w, paper eq. (7)."""
    v = jnp.asarray(v)
    w = jnp.asarray(w)
    nv = jnp.linalg.norm(v)
    nw = jnp.linalg.norm(w)
    denom = jnp.maximum(nv * nw, eps)
    cos_th = jnp.clip(jnp.dot(v, w) / denom, -1.0, 1.0)
    return jnp.arccos(cos_th)


def oriented_plane_basis(
    v: Array, w: Array, eps: float = _EPS
) -> tuple[Array, Array, Array]:
    """Orthonormal (q1, q2) spanning span{v, w}, oriented so q1 || v and
    q2 points along the component of w orthogonal to v.

    Rotating q1 toward q2 is then rotating v toward w. Returns
    (q1, q2, valid) where valid is false if the plane is degenerate.
    """
    v = jnp.asarray(v)
    w = jnp.asarray(w)
    nv = jnp.linalg.norm(v)
    nw = jnp.linalg.norm(w)
    q1 = v / jnp.maximum(nv, eps)
    w_orth = w - jnp.dot(q1, w) * q1
    n_orth = jnp.linalg.norm(w_orth)
    q2 = w_orth / jnp.maximum(n_orth, eps)
    valid = (nv > eps) & (nw > eps) & (n_orth > eps)
    return q1, q2, valid


def _plane_rotation_parts(
    v: Array, w: Array, alpha: Array | float, eps: float = _EPS
) -> tuple[Array, Array, Array]:
    """Return (P, R_phi - I_2, valid) for the 2D plane rotation."""
    q1, q2, valid = oriented_plane_basis(v, w, eps=eps)
    P = jnp.stack([q1, q2], axis=1)
    theta = angle_between(v, w, eps=eps)
    phi = jnp.asarray(alpha) * theta
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    R = jnp.array([[c, -s], [s, c]], dtype=P.dtype)
    R_minus_I = R - jnp.eye(2, dtype=P.dtype)
    return P, R_minus_I, valid


def rotation_matrix(v: Array, w: Array, alpha: Array | float, eps: float = _EPS) -> Array:
    """Dense M x M rotation A = Q B_phi Q^T, paper eqs. (7)-(9).

    Built as I + P (R_phi - I_2) P^T so the plane is oriented v -> w
    without a full QR of an M x M matrix. For tests and tiny M only.
    Degenerate (v, w) returns I.
    """
    v = jnp.asarray(v)
    w = jnp.asarray(w)
    M = v.shape[0]
    P, R_minus_I, valid = _plane_rotation_parts(v, w, alpha, eps=eps)
    A = jnp.eye(M, dtype=v.dtype) + P @ (R_minus_I @ P.T)
    return jnp.where(valid, A, jnp.eye(M, dtype=v.dtype))


def apply_plane_rotation(
    Gamma: Array,
    v: Array,
    w: Array,
    alpha: Array | float,
    eps: float = _EPS,
) -> Array:
    """Apply the plane rotation to Gamma without forming A.

    A Gamma = Gamma + P (R_phi - I_2) (P^T Gamma), cost O(M d).
    """
    Gamma = jnp.asarray(Gamma)
    v = jnp.asarray(v)
    w = jnp.asarray(w)
    P, R_minus_I, valid = _plane_rotation_parts(v, w, alpha, eps=eps)
    rotated = Gamma + P @ (R_minus_I @ (P.T @ Gamma))
    return jnp.where(valid, rotated, Gamma)


def type_i_vectors(h: Array, mu: Array, Gamma: Array) -> tuple[Array, Array]:
    """Type I pair (v, w), paper eq. (10): w = h - mu, v = Gamma Gamma^T w."""
    h = jnp.asarray(h)
    mu = jnp.asarray(mu)
    Gamma = jnp.asarray(Gamma)
    w = h - mu
    v = Gamma @ (Gamma.T @ w)
    return v, w


def type_ii_index(w: Array, Gamma: Array) -> Array:
    """Index i* of the Gamma column maximizing (w^T gamma_i) / ||gamma_i||."""
    w = jnp.asarray(w)
    Gamma = jnp.asarray(Gamma)
    dots = Gamma.T @ w
    norms = jnp.linalg.norm(Gamma, axis=0)
    cosines = dots / jnp.maximum(norms, _EPS)
    return jnp.argmax(cosines)


def type_ii_vector(w: Array, Gamma: Array) -> Array:
    """Type II vector v = gamma_{i*}, paper eq. (11)."""
    Gamma = jnp.asarray(Gamma)
    i_star = type_ii_index(w, Gamma)
    return Gamma[:, i_star]


def rotate_type_i(
    Gamma: Array, h: Array, mu: Array, alpha: Array | float
) -> Array:
    """Type I update Gamma <- A Gamma with v = P w, w = h - mu."""
    v, w = type_i_vectors(h, mu, Gamma)
    return apply_plane_rotation(Gamma, v, w, alpha)


def rotate_type_ii(
    Gamma: Array, h: Array, mu: Array, alpha: Array | float
) -> Array:
    """Type II update: rotate the nearest basis column toward w = h - mu."""
    w = jnp.asarray(h) - jnp.asarray(mu)
    v = type_ii_vector(w, Gamma)
    return apply_plane_rotation(Gamma, v, w, alpha)
