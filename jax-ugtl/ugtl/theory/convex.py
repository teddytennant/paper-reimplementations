"""Projected gradient descent (Algorithm 3) and the setting of Theorem 6.

The domain is a box, which keeps the projection a clip and, for a diagonal
quadratic, makes the constrained optimum available in closed form. That matters
because the Theorem 6 bound is additive against min f, so a sloppy optimum would
turn the test into a check of the reference solver.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

__all__ = ["BoxQuadratic", "projected_gradient_descent", "adversarial_gradient_error"]


class BoxQuadratic:
    """f(x) = 0.5 sum_i a_i x_i^2 + b . x on the box [lo, hi], with a_i > 0.

    Convex and continuously differentiable everywhere, so Definition 9 holds with
    diameter D = ||hi - lo|| and gradient bound G = max_{x in X} ||grad f(x)||.
    Both constants are exact here, not estimates: the function separates, so the
    gradient norm is maximized by taking each coordinate to whichever endpoint
    makes |a_i x_i + b_i| larger, and the minimizer is the unconstrained one
    clipped to the box.
    """

    def __init__(self, a: jnp.ndarray, b: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray):
        self.a = jnp.asarray(a, dtype=jnp.float32)
        self.b = jnp.asarray(b, dtype=jnp.float32)
        self.lo = jnp.asarray(lo, dtype=jnp.float32)
        self.hi = jnp.asarray(hi, dtype=jnp.float32)
        if jnp.any(self.a <= 0):
            raise ValueError("the quadratic must be strictly convex, so a > 0")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(self.a * x**2) + jnp.sum(self.b * x)

    def grad(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.a * x + self.b

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pi_X, the Euclidean projection onto the box."""
        return jnp.clip(x, self.lo, self.hi)

    @property
    def diameter(self) -> float:
        return float(jnp.linalg.norm(self.hi - self.lo))

    @property
    def grad_bound(self) -> float:
        at_lo = jnp.abs(self.a * self.lo + self.b)
        at_hi = jnp.abs(self.a * self.hi + self.b)
        return float(jnp.linalg.norm(jnp.maximum(at_lo, at_hi)))

    def argmin(self) -> jnp.ndarray:
        return self.project(-self.b / self.a)

    def min_value(self) -> float:
        return float(self(self.argmin()))


def adversarial_gradient_error(f: BoxQuadratic, zeta_g: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """A surrogate gradient obeying Definition 5 with tolerance zeta_g.

    The error term has length exactly zeta_g and points from the optimum towards
    the current iterate, so the update it induces is pushed away from the
    optimum. Definition 5 then holds at every queried point, with equality, which
    is the case Theorem 6 is stated against.
    """

    x_star = f.argmin()

    def grad_tilde(x: jnp.ndarray) -> jnp.ndarray:
        direction = x - x_star
        norm = jnp.linalg.norm(direction)
        unit = jnp.where(norm > 1e-12, direction / jnp.maximum(norm, 1e-12), jnp.ones_like(x) / jnp.sqrt(x.size))
        return f.grad(x) - zeta_g * unit

    return grad_tilde


def projected_gradient_descent(
    f: BoxQuadratic,
    grad_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x1: jnp.ndarray,
    eta: float,
    T: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Algorithm 3 run on a surrogate gradient.

    Line 3 is x_{t+1} = Pi_X[x_t - eta grad f~(x_t)] and line 5 returns the mean
    of the first T iterates. Returns that mean plus the full trajectory, since
    Definition 5 is a statement about the points where the algorithm queried a
    gradient.
    """

    def step(x, _):
        x_next = f.project(x - eta * grad_fn(x))
        return x_next, x

    x_last, traj = jax.lax.scan(step, f.project(x1), None, length=T)
    return jnp.mean(traj, axis=0), traj
