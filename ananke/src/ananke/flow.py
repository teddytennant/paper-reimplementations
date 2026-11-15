"""Exact circular reaction flow (Branch A, Theorem 1).

The map is the closed-form solution of

    r' = -beta * r * log(r / R)
    z' = -beta * z,  z = log(r / R)
    theta' = alpha_eff + a * z

on the punctured plane R^2 \\ {c}. Parameters are frozen over the step.
The primary path is the exact trigonometric rotation. Appendix E dual-shear
is available as an optional algebraic substitute and is not used by default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rotation_matrix(theta):
    """Canonical planar rotation R_theta in SO(2).

    [[cos theta, -sin theta],
     [sin theta,  cos theta]]
    """
    theta = jnp.asarray(theta)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    row0 = jnp.stack([cos_t, -sin_t], axis=-1)
    row1 = jnp.stack([sin_t, cos_t], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def apply_rotation(v, theta):
    """Apply R_theta to vectors v of shape (..., 2)."""
    rot = rotation_matrix(theta)
    return jnp.matmul(rot, jnp.asarray(v)[..., None])[..., 0]


def apply_dual_shear(v, s):
    """Area-preserving dual-shear M(s) from Appendix E, s = theta.

    Hardware form (two FMAs), equivalent to

        M(s) = [[1, -s], [0, 1]] @ [[1, 0], [s, 1]]
             = [[1 - s^2, -s], [s, 1]]

    det M(s) = 1. This is not a Euclidean rotation: it preserves a tilted
    elliptic metric, not the circle, and is not the primary path.
    """
    v = jnp.asarray(v)
    s = jnp.asarray(s, dtype=v.dtype)
    x0 = v[..., 0]
    y0 = v[..., 1]
    y_mid = y0 + s * x0
    x_p = x0 - s * y_mid
    y_p = y_mid
    return jnp.stack([x_p, y_p], axis=-1)


def _as_float(x):
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.float64)
    return x


def flow_quantities(x, c, R, beta, alpha_eff, a, t, eps=0.0, rotation="exact"):
    """Closed-form state of the exact circular flow.

    x: (..., 2) initial point. Must not equal c when eps is 0.
    c: broadcastable to x.
    R, beta, alpha_eff, a, t: broadcastable to x.shape[:-1]. R > 0, beta > 0, t >= 0.
    eps: optional floor on the radius, used by the network path only.
    rotation: "exact" (SO(2), primary) or "dual_shear" (Appendix E).

    Returns a dict with z0, q, z, r, theta, rho, r0, y.
    z(t) = q * z0, r(t) = R * exp(q * z0),
    theta(t) = alpha_eff * t + (a / beta) * (1 - q) * z0,
    rho = (R / r0) ** (1 - q),
    y = Phi_t(x) = c + rho * R_theta (x - c).
    """
    if rotation not in ("exact", "dual_shear"):
        raise ValueError("rotation must be 'exact' or 'dual_shear'")

    x = _as_float(x)
    dtype = x.dtype
    c = jnp.asarray(c, dtype=dtype)
    R = jnp.asarray(R, dtype=dtype)
    beta = jnp.asarray(beta, dtype=dtype)
    alpha_eff = jnp.asarray(alpha_eff, dtype=dtype)
    a = jnp.asarray(a, dtype=dtype)
    t = jnp.asarray(t, dtype=dtype)

    delta = x - c
    r0 = jnp.linalg.norm(delta, axis=-1)
    r0_used = jnp.maximum(r0, jnp.asarray(eps, dtype=dtype))
    z0 = jnp.log(r0_used / R)
    q = jnp.exp(-beta * t)
    z = q * z0
    r = R * jnp.exp(q * z0)
    theta = alpha_eff * t + (a / beta) * (1.0 - q) * z0
    # Equivalent to (R / r0) ** (1 - q) for R, r0 > 0, and exact at t = 0.
    rho = jnp.exp((1.0 - q) * jnp.log(R / r0_used))

    if rotation == "exact":
        rotated = apply_rotation(delta, theta)
    else:
        rotated = apply_dual_shear(delta, theta)
    y = c + rho[..., None] * rotated

    return {
        "z0": z0,
        "q": q,
        "z": z,
        "r": r,
        "theta": theta,
        "rho": rho,
        "r0": r0_used,
        "y": y,
    }


def exact_circular_flow(x, c, R, beta, alpha_eff, a, t, eps=0.0, rotation="exact"):
    """Phi_t(x) for one 2D phase plane. See flow_quantities."""
    return flow_quantities(
        x, c, R, beta, alpha_eff, a, t, eps=eps, rotation=rotation
    )["y"]


def product_flow(x, c, R, beta, alpha_eff, a, t, eps=0.0, rotation="exact"):
    """Apply the circular flow independently on consecutive channel pairs.

    x has shape (..., 2K). Scalar parameters are shared across planes.
    Per-plane arrays broadcast against (*batch, K): c may be (2,), (K, 2),
    or (..., K, 2); R, beta, alpha_eff, a, t may be scalar or (..., K).
    """
    x = _as_float(x)
    if x.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even (D = 2K phase planes)")
    k = x.shape[-1] // 2
    lead = x.shape[:-1]
    planes = x.reshape(*lead, k, 2)
    y = exact_circular_flow(
        planes, c, R, beta, alpha_eff, a, t, eps=eps, rotation=rotation
    )
    return y.reshape(*lead, 2 * k)


def integrate_polar_rk4(r0, theta0, R, beta, alpha_eff, a, t, dt):
    """Fixed-step RK4 for the polar ODE of Theorem 1.

    r' = -beta * r * log(r / R)
    theta' = alpha_eff + a * log(r / R)

    t and dt are concrete numbers. The step is resized so n * dt_eff = t.
    Returns (r(t), theta(t)) with theta the absolute angle, starting at theta0.
    This integrator is a numerical check only. The network uses the closed form.
    """
    t = float(t)
    dt = float(dt)
    if t < 0.0 or dt <= 0.0:
        raise ValueError("t must be >= 0 and dt must be > 0")

    r0 = _as_float(r0)
    dtype = r0.dtype
    theta0 = jnp.asarray(theta0, dtype=dtype)
    R = jnp.asarray(R, dtype=dtype)
    beta = jnp.asarray(beta, dtype=dtype)
    alpha_eff = jnp.asarray(alpha_eff, dtype=dtype)
    a = jnp.asarray(a, dtype=dtype)

    if t == 0.0:
        return r0, theta0

    n = max(int(round(t / dt)), 1)
    dt_eff = jnp.asarray(t / n, dtype=dtype)

    def f(state):
        r = state[0]
        z = jnp.log(r / R)
        return jnp.stack([-beta * r * z, alpha_eff + a * z])

    def step(state, _):
        k1 = f(state)
        k2 = f(state + 0.5 * dt_eff * k1)
        k3 = f(state + 0.5 * dt_eff * k2)
        k4 = f(state + dt_eff * k3)
        nxt = state + (dt_eff / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return nxt, None

    state0 = jnp.stack([r0, theta0])
    state, _ = jax.lax.scan(step, state0, None, length=n)
    return state[0], state[1]


def cartesian_from_polar(c, r, theta):
    """Reconstruct c + r * (cos theta, sin theta)."""
    c = jnp.asarray(c)
    r = jnp.asarray(r, dtype=c.dtype)
    theta = jnp.asarray(theta, dtype=c.dtype)
    offset = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)], axis=-1)
    return c + offset


def flow_vs_rk4_error(x, c, R, beta, alpha_eff, a, t, dt):
    """Max-abs Cartesian error between the closed form and polar RK4."""
    x = _as_float(x)
    dtype = x.dtype
    c = jnp.asarray(c, dtype=dtype)
    y = exact_circular_flow(x, c, R, beta, alpha_eff, a, t)
    delta = x - c
    r0 = jnp.linalg.norm(delta)
    theta0 = jnp.arctan2(delta[1], delta[0])
    r_n, theta_n = integrate_polar_rk4(r0, theta0, R, beta, alpha_eff, a, t, dt)
    y_n = cartesian_from_polar(c, r_n, theta_n)
    return jnp.max(jnp.abs(y - y_n))
