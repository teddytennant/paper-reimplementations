"""Cubic B-spline trajectories (eq 2.5–2.6) via Cox–de Boor evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


SPLINE_DEGREE = 3


def clamped_knot_vector(t_max: float, n_interior: int, degree: int = SPLINE_DEGREE) -> Array:
    """Equally spaced interior knots on [0, T] with endpoint multiplicity p+1.

    For cubic (p=3) B-splines with K coefficients, n_interior = K - p - 1.
    """
    t_max = float(t_max)
    n_interior = int(n_interior)
    degree = int(degree)
    if n_interior < 0:
        raise ValueError("n_interior must be nonnegative")
    if n_interior == 0:
        interior = jnp.array([], dtype=jnp.float64)
    else:
        interior = jnp.linspace(0.0, t_max, n_interior + 2, dtype=jnp.float64)[1:-1]
    left = jnp.zeros(degree + 1, dtype=jnp.float64)
    right = jnp.full(degree + 1, t_max, dtype=jnp.float64)
    return jnp.concatenate([left, interior, right])


def n_basis(knots: Array, degree: int = SPLINE_DEGREE) -> int:
    """Number of B-spline basis functions: K = n_knots − p − 1."""
    return int(knots.shape[0] - degree - 1)


def find_span(t: Array, knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """Knot span index i with knots[i] <= t < knots[i+1] (last span includes T)."""
    p = int(degree)
    n = knots.shape[0] - p - 2
    t_max = knots[n + 1]
    t = jnp.asarray(t, dtype=jnp.float64)
    i = jnp.searchsorted(knots, t, side="right") - 1
    i = jnp.clip(i, p, n)
    return jnp.where(t >= t_max, n, i).astype(jnp.int32)


def _basis_funs(span: Array, t: Array, knots: Array, degree: int) -> Array:
    """Nonzero degree-p basis values at t: length p+1, starting at span−p."""
    p = int(degree)
    nvals = jnp.zeros(p + 1, dtype=jnp.float64)
    nvals = nvals.at[0].set(1.0)
    left = jnp.zeros(p + 1, dtype=jnp.float64)
    right = jnp.zeros(p + 1, dtype=jnp.float64)
    for j in range(1, p + 1):
        left = left.at[j].set(t - knots[span + 1 - j])
        right = right.at[j].set(knots[span + j] - t)
        saved = jnp.array(0.0, dtype=jnp.float64)
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            tmp = jnp.where(denom > 0.0, nvals[r] / denom, 0.0)
            nvals = nvals.at[r].set(saved + right[r + 1] * tmp)
            saved = left[j - r] * tmp
        nvals = nvals.at[j].set(saved)
    return nvals


def basis_vector(t: Array, knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """φ(t) ∈ R^K, the vector of B-spline basis functions at t (eq 2.5)."""
    t = jnp.asarray(t, dtype=jnp.float64)
    knots = jnp.asarray(knots, dtype=jnp.float64)
    p = int(degree)
    k = n_basis(knots, p)
    span = find_span(t, knots, p)
    local = _basis_funs(span, t, knots, p)
    phi = jnp.zeros(k, dtype=jnp.float64)
    start = span - p
    for j in range(p + 1):
        phi = phi.at[start + j].set(local[j])
    return phi


def basis_matrix(times: Array, knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """Stack φ(t_m)^T as rows: shape (M, K)."""
    times = jnp.asarray(times, dtype=jnp.float64)
    return jnp.stack([basis_vector(t, knots, degree) for t in times])


def basis_derivative(t: Array, knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """φ'(t) from the standard B-spline derivative recurrence."""
    t = jnp.asarray(t, dtype=jnp.float64)
    knots = jnp.asarray(knots, dtype=jnp.float64)
    p = int(degree)
    k = n_basis(knots, p)
    if p == 0:
        return jnp.zeros(k, dtype=jnp.float64)
    span = find_span(t, knots, p - 1)
    lower = _basis_funs(span, t, knots, p - 1)
    n_lower = n_basis(knots, p - 1)
    n_low = jnp.zeros(n_lower, dtype=jnp.float64)
    start = span - (p - 1)
    for j in range(p):
        n_low = n_low.at[start + j].set(lower[j])
    dphi = jnp.zeros(k, dtype=jnp.float64)
    for i in range(k):
        d1 = knots[i + p] - knots[i]
        d2 = knots[i + p + 1] - knots[i + 1]
        term1 = jnp.where(d1 > 0.0, (p / d1) * n_low[i], 0.0)
        term2 = jnp.where(d2 > 0.0, (p / d2) * n_low[i + 1], 0.0)
        dphi = dphi.at[i].set(term1 - term2)
    return dphi


def basis_second_derivative(t: Array, knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """φ''(t) by differentiating the degree-(p−1) formula."""
    t = jnp.asarray(t, dtype=jnp.float64)
    knots = jnp.asarray(knots, dtype=jnp.float64)
    p = int(degree)
    k = n_basis(knots, p)
    d_lower = basis_derivative(t, knots, p - 1)
    d2 = jnp.zeros(k, dtype=jnp.float64)
    for i in range(k):
        d1 = knots[i + p] - knots[i]
        d2n = knots[i + p + 1] - knots[i + 1]
        term1 = jnp.where(d1 > 0.0, (p / d1) * d_lower[i], 0.0)
        term2 = jnp.where(d2n > 0.0, (p / d2n) * d_lower[i + 1], 0.0)
        d2 = d2.at[i].set(term1 - term2)
    return d2


def greville_abscissae(knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """Greville sites ξ_i = (t_{i+1} + ⋯ + t_{i+p}) / p; coefficients of f(t)=t."""
    p = int(degree)
    k = n_basis(knots, p)
    return jnp.stack([jnp.mean(knots[i + 1 : i + p + 1]) for i in range(k)])


def evaluate_curve(coeffs: Array, times: Array, knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """Scalar spline values at times; coeffs shape (K,)."""
    phi = basis_matrix(times, knots, degree)
    return phi @ coeffs


def evaluate_trajectories(
    c_x: Array,
    c_y: Array,
    times: Array,
    knots: Array,
    degree: int = SPLINE_DEGREE,
) -> Array:
    """z_i(t) = (c_i^x · φ(t), c_i^y · φ(t)), shape (n, M, 2) (eq 2.5)."""
    phi = basis_matrix(times, knots, degree)
    x = c_x @ phi.T
    y = c_y @ phi.T
    return jnp.stack([x, y], axis=-1)


def squared_distance_quadratic(
    c_i_x: Array,
    c_j_x: Array,
    c_i_y: Array,
    c_j_y: Array,
    phi_t: Array,
) -> Array:
    """d_ij(t)² from the coefficient quadratic form (eq 2.6).

    d_ij(t)² = (c_i^x − c_j^x)^T φ(t)φ(t)^T (c_i^x − c_j^x)
             + (c_i^y − c_j^y)^T φ(t)φ(t)^T (c_i^y − c_j^y)
    """
    dx = c_i_x - c_j_x
    dy = c_i_y - c_j_y
    outer = jnp.outer(phi_t, phi_t)
    return dx @ outer @ dx + dy @ outer @ dy


def gram_phi(phi_t: Array) -> Array:
    """φ(t) φ(t)^T."""
    return jnp.outer(phi_t, phi_t)


@dataclass(frozen=True)
class SplineBasis:
    """Clamped cubic B-spline basis on [0, T] with equally spaced interior knots."""

    knots: Array
    degree: int
    t_max: float

    @property
    def n_coeff(self) -> int:
        return n_basis(self.knots, self.degree)

    def phi(self, t: Array) -> Array:
        return basis_vector(t, self.knots, self.degree)

    def phi_matrix(self, times: Array) -> Array:
        return basis_matrix(times, self.knots, self.degree)


def make_basis(t_max: float, n_coeff: int, degree: int = SPLINE_DEGREE) -> SplineBasis:
    """Build a clamped cubic basis with K = n_coeff coefficients."""
    n_interior = n_coeff - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_coeff must be at least degree+1={degree + 1}")
    knots = clamped_knot_vector(t_max, n_interior, degree)
    return SplineBasis(knots=knots, degree=degree, t_max=float(t_max))


def fit_coefficients(values: Array, times: Array, knots: Array, degree: int = SPLINE_DEGREE) -> Array:
    """Least-squares B-spline coefficients for a scalar function sampled at times."""
    phi = basis_matrix(times, knots, degree)
    coeffs, *_ = jnp.linalg.lstsq(phi, values, rcond=None)
    return coeffs
