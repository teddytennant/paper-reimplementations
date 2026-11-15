"""Fixed-grid quadrature helpers.  Pure jax primitives, no adaptive solver."""

import jax.numpy as jnp


def simpson_weights(n: int, h: float):
    """Composite Simpson weights on n uniformly spaced nodes.  n must be odd."""
    if n % 2 == 0:
        raise ValueError("Simpson needs an odd number of nodes, got %d" % n)
    w = jnp.ones(n)
    w = w.at[1:-1:2].set(4.0)
    w = w.at[2:-1:2].set(2.0)
    return w * (h / 3.0)


def cum_quad4(f, h):
    """Cumulative integral of f on a uniform grid, fourth order.

    Returns I with I[0] = 0 and I[k] approximating int_{x_0}^{x_k} f.  Each
    panel uses the four-point Adams-Moulton rule, so the error is O(h^4)
    globally instead of the O(h^2) a cumulative trapezoid would give.  We need
    the cumulative form, not a single total, because the profile is stored as a
    monotone map and every node of it has to be right.
    """
    n = f.shape[0]
    if n < 4:
        raise ValueError("cum_quad4 needs at least 4 nodes")
    # interior panels [x_j, x_{j+1}] for j = 1 .. n-3
    mid = (-f[:-3] + 13.0 * f[1:-2] + 13.0 * f[2:-1] - f[3:]) / 24.0
    first = (9.0 * f[0] + 19.0 * f[1] - 5.0 * f[2] + f[3]) / 24.0
    last = (9.0 * f[-1] + 19.0 * f[-2] - 5.0 * f[-3] + f[-4]) / 24.0
    panels = jnp.concatenate(
        [jnp.atleast_1d(first), mid, jnp.atleast_1d(last)]
    )
    return h * jnp.concatenate([jnp.zeros(1), jnp.cumsum(panels)])


def hermite_interp(xs, ys, dys, x):
    """Cubic Hermite interpolation of a table with known slopes.

    xs must be strictly increasing, ys the values and dys = dy/dx at the nodes.
    Having the exact slopes on hand is the whole point: the profile solver knows
    dy/du = -xi(exp(y)) at every node from the ODE itself, so the interpolant is
    fourth order in the node spacing and its first derivative third order, which
    is what the zero-flux residual check leans on.  Outside the table it falls
    back to a straight line off the nearest end node.
    """
    n = xs.shape[0]
    i = jnp.clip(jnp.searchsorted(xs, x, side="right") - 1, 0, n - 2)
    x0 = xs[i]
    x1 = xs[i + 1]
    h = x1 - x0
    t = (x - x0) / h
    tc = jnp.clip(t, 0.0, 1.0)
    t2 = tc * tc
    t3 = t2 * tc
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + tc
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    inside = h00 * ys[i] + h10 * h * dys[i] + h01 * ys[i + 1] + h11 * h * dys[i + 1]
    end_slope = jnp.where(t < 0.0, dys[i], dys[i + 1])
    return inside + (t - tc) * h * end_slope
