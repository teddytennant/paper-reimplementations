"""Simplex allocation by dual bisection on separable marginal costs (JAX)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

_EPS = 1e-06


def entropy(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    pos = x > 0
    safe = jnp.where(pos, x, 1.0)
    return jnp.sum(jnp.where(pos, x * jnp.log(safe), 0.0))


def entropy_pcsp(x):
    return jnp.asarray(x, dtype=jnp.float32)


_HINGE_BETA = 32.0


def hinge_residual(x, eta, knots):
    """Smooth hinge: softplus(eta) * softplus(beta * (x - knot)) / beta."""
    x = jnp.asarray(x, dtype=jnp.float32)
    weights = jax.nn.softplus(jnp.asarray(eta, dtype=jnp.float32))
    knots = jnp.asarray(knots, dtype=jnp.float32)
    beta = jnp.float32(_HINGE_BETA)
    slack = jax.nn.softplus(beta * (x[..., None] - knots)) / beta
    return jnp.sum(weights * slack)


def _hinge_grad(x, eta, knots):
    """d/dx hinge_residual. Matches jax.grad of the smooth hinge."""
    x = jnp.asarray(x, dtype=jnp.float32)
    weights = jax.nn.softplus(jnp.asarray(eta, dtype=jnp.float32))
    knots = jnp.asarray(knots, dtype=jnp.float32)
    beta = jnp.float32(_HINGE_BETA)
    sig = jax.nn.sigmoid(beta * (x[..., None] - knots))
    return jnp.sum(weights * sig, axis=-1)


def quadratic_path(B):
    """Minimizer for H=[[2,3],[3,5]], a=(0,1), I=a: (B, 0) on [0, 2]."""
    b = jnp.asarray(B, dtype=jnp.float32)
    zero = jnp.zeros_like(b)
    if b.ndim == 0:
        return (float(b), float(zero))
    return (b, zero)


def quadratic_H_inv_ones():
    """H^{-1} 1 for H = [[2, 3], [3, 5]]. Exact value is (2, -1)."""
    H = jnp.array([[2.0, 3.0], [3.0, 5.0]], dtype=jnp.float32)
    solved = jnp.linalg.solve(H, jnp.ones((2,), dtype=jnp.float32))
    snapped = jnp.rint(solved)
    return (int(snapped[0]), int(snapped[1]))


def quadratic_allocate(I, B):
    """2-d quadratic allocation with H=[[2, 3], [3, 5]], a=(0, 1), c=0.

    min_{z >= 0, sum z = B} 0.5 * (I + z - a)^T H (I + z - a).
    Sum-constrained critical point x = a + nu * H^{-1} 1, with
    nu = 1^T I + B - 1^T a, then projected onto z >= 0.
    """
    I = jnp.asarray(I, dtype=jnp.float64)
    B = jnp.asarray(B, dtype=jnp.float64)
    a = jnp.array([0.0, 1.0], dtype=jnp.float64)
    h_inv_ones = jnp.array(quadratic_H_inv_ones(), dtype=jnp.float64)
    nu = jnp.sum(I) + B - jnp.sum(a)
    z = a + nu * h_inv_ones - I
    z = jnp.where(z[0] < 0.0, jnp.stack([jnp.array(0.0, dtype=jnp.float64), B]), z)
    z = jnp.where(z[1] < 0.0, jnp.stack([B, jnp.array(0.0, dtype=jnp.float64)]), z)
    return z


def _slopes(eta, knots, n, dtype):
    eta = jnp.asarray(eta, dtype=dtype)
    knots = jnp.asarray(knots, dtype=dtype)
    weights = jax.nn.softplus(eta)
    if weights.ndim == 0:
        weights = jnp.broadcast_to(weights, (n, knots.shape[0]))
    elif weights.ndim == 1:
        if weights.shape[0] == n and knots.shape[0] != n:
            weights = weights[:, None]
        else:
            weights = jnp.broadcast_to(weights, (n, knots.shape[0]))
    return weights, knots


@partial(jax.jit, static_argnames=("steps",))
def allocate(I, c, B, tau, eta, knots, steps=48):
    """Budget-B allocation by bisection on the separable marginal cost.

    m_i(x) = c_i + tau * (log(x) + 1) + d/dx_i hinge_residual(x, eta, knots).

    Dual nu is bisected so sum_i relu(x_i(nu) - I_i) = B. The hinge is smooth,
    so the marginal the dual equalizes is the one jax.grad reports.
    """
    del steps  # retained so existing call sites stay valid; search length is fixed
    I = jnp.asarray(I, dtype=jnp.float32)
    c = jnp.asarray(c, dtype=jnp.float32)
    B = jnp.asarray(B, dtype=jnp.float32)
    tau = jnp.asarray(tau, dtype=jnp.float32)
    eta = jnp.asarray(eta, dtype=jnp.float32)
    knots = jnp.atleast_1d(jnp.asarray(knots, dtype=jnp.float32))

    x_lo = jnp.float32(1e-4)
    x_hi = jnp.maximum(I + B + jnp.float32(1.0), x_lo + jnp.float32(1e-3))

    def marginal(x):
        return c + tau * (jnp.log(x) + jnp.float32(1.0)) + _hinge_grad(x, eta, knots)

    def invert_marginal(nu):
        lo = jnp.full_like(I, x_lo)
        hi = x_hi

        def body(_, bounds):
            lo_b, hi_b = bounds
            mid = (lo_b + hi_b) * jnp.float32(0.5)
            go_hi = marginal(mid) < nu
            lo_b = jnp.where(go_hi, mid, lo_b)
            hi_b = jnp.where(go_hi, hi_b, mid)
            return lo_b, hi_b

        lo, hi = jax.lax.fori_loop(0, 40, body, (lo, hi))
        return (lo + hi) * jnp.float32(0.5)

    def excess(nu):
        x = invert_marginal(nu)
        return jnp.sum(jnp.maximum(x - I, jnp.float32(0.0))) - B

    nu_lo = jnp.min(marginal(jnp.full_like(I, x_lo))) - jnp.float32(1.0)
    nu_hi = jnp.max(marginal(x_hi)) + jnp.float32(1.0)

    def bisect_nu(_, bounds):
        lo_b, hi_b = bounds
        mid = (lo_b + hi_b) * jnp.float32(0.5)
        go_hi = excess(mid) < jnp.float32(0.0)
        lo_b = jnp.where(go_hi, mid, lo_b)
        hi_b = jnp.where(go_hi, hi_b, mid)
        return lo_b, hi_b

    nu_lo, nu_hi = jax.lax.fori_loop(0, 48, bisect_nu, (nu_lo, nu_hi))
    nu = (nu_lo + nu_hi) * jnp.float32(0.5)
    x = invert_marginal(nu)
    return jnp.maximum(x - I, jnp.float32(0.0))

