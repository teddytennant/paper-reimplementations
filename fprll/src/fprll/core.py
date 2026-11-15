"""Box domain, separable leader, closed-form density, grid LazySample."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Uniform midpoint grid on (-1, 1). Documented resolution for sampling and TV.
GRID_N = 1024
BOX_LOW = -1.0
BOX_HIGH = 1.0


def make_grid(n: int = GRID_N, low: float = BOX_LOW, high: float = BOX_HIGH):
    """Midpoints and cell width of n equal bins on [low, high]."""
    n = int(n)
    dx = (high - low) / n
    centers = low + (np.arange(n, dtype=np.float64) + 0.5) * dx
    return centers, dx


def decision_bin(x, n_bins: int = GRID_N, low: float = BOX_LOW, high: float = BOX_HIGH) -> int:
    """Bin index of a scalar decision on the uniform grid."""
    dx = (high - low) / int(n_bins)
    i = int(math.floor((float(x) - low) / dx))
    if i < 0:
        return 0
    if i >= n_bins:
        return n_bins - 1
    return i


def indicator_cost(x, y, n_bins: int = GRID_N) -> int:
    """1 iff consecutive grid-binned decisions differ, else 0."""
    xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
    ya = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if xa.shape != ya.shape:
        raise ValueError("indicator_cost expects decisions of the same shape")
    bx = [decision_bin(float(v), n_bins) for v in xa.ravel()]
    by = [decision_bin(float(v), n_bins) for v in ya.ravel()]
    return int(any(a != b for a, b in zip(bx, by)))


def _as_f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def potential_grad(x, cum_grad, gamma: float, sigma: float):
    """Gradient of unperturbed F(x) = <cum_grad, x> + (1/gamma) r(x).

    r(x) = (sigma/2) ||x||^2 - sum_i log(1 - x_i^2). The perturbation is not
    included; the leader minimizer solves grad F(x) = Z.
    """
    xx = _as_f64(x)
    gg = _as_f64(cum_grad)
    barrier = sigma * xx + 2.0 * xx / (1.0 - xx * xx)
    return np.asarray(gg + (1.0 / gamma) * barrier, dtype=np.float64)


def potential_hess(x, cum_grad, gamma: float, sigma: float):
    """Diagonal Hessian of the unperturbed leader. cum_grad does not enter."""
    del cum_grad
    xx = _as_f64(x)
    numer = 2.0 * (1.0 + xx * xx)
    denom = (1.0 - xx * xx) ** 2
    hess = (1.0 / gamma) * (sigma + numer / denom)
    return np.asarray(hess, dtype=np.float64)


def _laplace_scale(laplace_scale: float) -> float:
    """Paper's Laplace scale mu. Not a cost weight."""
    mu = float(laplace_scale)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("laplace_scale must be positive")
    return mu


def _log_density_parts(x, cum_grad, gamma, sigma, laplace_scale):
    xx = _as_f64(x)
    gg = _as_f64(cum_grad)
    barrier_g = sigma * xx + 2.0 * xx / (1.0 - xx * xx)
    grad = gg + (1.0 / gamma) * barrier_g
    hess = (1.0 / gamma) * (sigma + 2.0 * (1.0 + xx * xx) / (1.0 - xx * xx) ** 2)
    # Q = nu(-grad F) * abs(det(-hess F)). Hessian is diagonal, so the
    # determinant factor is the product of abs(-hess_ii).
    # p = -grad F ~ Lap(mu), mu = laplace_scale:
    # log nu(p) = -d*log(2*mu) - ||p||_1 / mu.
    # Per coordinate this is -log(2*mu) - |p_i| / mu; the sum over coordinates
    # is the d-dimensional formula.
    mu = _laplace_scale(laplace_scale)
    p = -grad
    log_nu = -jnp.log(2.0 * mu) - jnp.abs(p) / mu
    log_factor = jnp.log(jnp.abs(-hess))
    return log_nu, log_factor, grad


def density(x, cum_grad, gamma: float, sigma: float, laplace_scale: float = 1.0):
    """Closed form Q(x) = nu(-grad F(x)) * abs(det(-hess F(x))).

    nu is Lap(mu) with mu = laplace_scale (the paper's mu, not a cost weight):
    log nu(p) = -d*log(2*mu) - ||p||_1 / mu, where p = -grad F.
    The separable 1D case is -log(2*mu) - |p| / mu.
    If cum_grad is scalar, each entry of x is a 1D point. If cum_grad has
    shape (d,), the last axis of x is the coordinate axis and the density is
    the separable product.
    """
    log_nu, log_factor, _grad = _log_density_parts(
        x, cum_grad, gamma, sigma, laplace_scale
    )
    c = _as_f64(cum_grad)
    if c.ndim == 0:
        return np.asarray(jnp.exp(log_nu + log_factor), dtype=np.float64)
    return np.asarray(jnp.exp(jnp.sum(log_nu + log_factor, axis=-1)), dtype=np.float64)


def log_density(x, cum_grad, gamma: float, sigma: float, laplace_scale: float = 1.0):
    """Log of density. Same broadcasting rules as density."""
    log_nu, log_factor, _grad = _log_density_parts(
        x, cum_grad, gamma, sigma, laplace_scale
    )
    c = _as_f64(cum_grad)
    if c.ndim == 0:
        return np.asarray(log_nu + log_factor, dtype=np.float64)
    return np.asarray(jnp.sum(log_nu + log_factor, axis=-1), dtype=np.float64)


def minimizer(cum_grad, perturbation, gamma: float, sigma: float, n_iter: int = 60):
    """Coordinate-wise bisection of argmin F(x) - <Z, x> on (-1, 1)^d.

    The derivative of each coordinate is strictly increasing, so bisection
    finds the unique critical point.
    """
    c = np.asarray(cum_grad, dtype=np.float64)
    z = np.asarray(perturbation, dtype=np.float64)
    shape = np.broadcast(c, z).shape
    lo = np.full(shape, -1.0 + 1e-12, dtype=np.float64)
    hi = np.full(shape, 1.0 - 1e-12, dtype=np.float64)
    c_b = np.broadcast_to(c, shape)
    z_b = np.broadcast_to(z, shape)
    inv_gamma = 1.0 / float(gamma)
    sig = float(sigma)
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        deriv = c_b + inv_gamma * (sig * mid + 2.0 * mid / (1.0 - mid * mid)) - z_b
        go_left = deriv > 0.0
        hi = np.where(go_left, mid, hi)
        lo = np.where(go_left, lo, mid)
    return 0.5 * (lo + hi)


def total_variation(p, q, dx: float) -> float:
    """Grid total variation 0.5 * sum |p - q| * dx of two density samples."""
    pp = np.asarray(p, dtype=np.float64)
    qq = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(pp - qq)) * float(dx))


def grid_mean(density_values, centers, dx: float) -> float:
    """Mean coordinate under a 1D grid density. Mass-normalized."""
    q = np.maximum(np.asarray(density_values, dtype=np.float64), 0.0)
    mass = float(np.sum(q) * dx)
    if mass <= 0.0:
        return 0.0
    return float(np.sum(q * np.asarray(centers, dtype=np.float64)) * dx / mass)


def sample_index(density_values, rng: np.random.Generator) -> int:
    """Categorical index with masses proportional to density values (uniform dx)."""
    w = np.maximum(np.asarray(density_values, dtype=np.float64), 0.0)
    total = float(w.sum())
    if total <= 0.0 or not np.isfinite(total):
        return int(rng.integers(len(w)))
    return int(rng.choice(len(w), p=w / total))


def lazy_sample(index: int, q, p, rng: np.random.Generator, max_tries: int = 10000):
    """alg:lazy-sample on a shared 1D grid.

    q and p are closed-form density values at the grid centers. Keep the
    current index with probability min(1, p/q) at that point, implemented as
    z ~ Unif[0, q(x)] and keep if z < p(x). Otherwise rejection-sample the
    residual, as written in the algorithm.
    Returns (new_index, kept).
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    idx = int(index)
    qx = float(q[idx])
    px = float(p[idx])
    if qx > 0.0 and np.isfinite(qx):
        z = float(rng.uniform(0.0, qx))
        if z < px:
            return idx, True
    weights = np.maximum(p, 0.0)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return idx, False
    cdf = np.cumsum(weights)
    for _ in range(int(max_tries)):
        j = int(np.searchsorted(cdf, float(rng.random()) * total, side="right"))
        if j >= len(p):
            j = len(p) - 1
        py = float(p[j])
        qy = float(q[j])
        if py <= 0.0 or not np.isfinite(py):
            continue
        z_prime = float(rng.uniform(0.0, py))
        if z_prime > qy:
            return j, False
    fallback = int(np.searchsorted(cdf, float(rng.random()) * total, side="right"))
    return min(fallback, len(p) - 1), False
