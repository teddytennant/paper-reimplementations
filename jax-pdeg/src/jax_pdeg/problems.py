"""Toy minimax problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from jax_pdeg.projections import project_unconstrained

Array = jnp.ndarray
Fn = Callable[[Array, Array], Array]
Proj = Callable[[Array], Array]


@dataclass
class MinimaxProblem:
    f: Fn
    grad_x: Fn
    grad_y: Fn
    project_x: Proj
    project_y: Proj


def make_problem(f: Fn, project_x: Proj | None = None, project_y: Proj | None = None) -> MinimaxProblem:
    if project_x is None:
        project_x = project_unconstrained
    if project_y is None:
        project_y = project_unconstrained
    return MinimaxProblem(
        f=f,
        grad_x=jax.grad(f, argnums=0),
        grad_y=jax.grad(f, argnums=1),
        project_x=project_x,
        project_y=project_y,
    )


def bilinear_saddle(A, mu: float = 0.0, project_x: Proj | None = None, project_y: Proj | None = None) -> MinimaxProblem:
    """f(x, y) = x^T A y - (μ/2) ||y||^2."""
    A = jnp.asarray(A, dtype=jnp.float64)

    def f(x, y):
        x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        return jnp.dot(x, A @ y) - 0.5 * mu * jnp.dot(y, y)

    return make_problem(f, project_x=project_x, project_y=project_y)


def quadratic_scsc(mu: float = 1.0, project_x: Proj | None = None, project_y: Proj | None = None) -> MinimaxProblem:
    """f(x, y) = (1/2)||x||^2 + x^T y - (μ/2)||y||^2. Saddle at the origin."""

    def f(x, y):
        x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        return 0.5 * jnp.dot(x, x) + jnp.dot(x, y) - 0.5 * mu * jnp.dot(y, y)

    return make_problem(f, project_x=project_x, project_y=project_y)


def ncc_interval(project_x: Proj | None = None, project_y: Proj | None = None) -> MinimaxProblem:
    """Nonconvex-concave toy f(x, y) = -x^2/4 + x y (scalars stored as 1-vectors)."""

    def f(x, y):
        x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        return -0.25 * x[0] ** 2 + x[0] * y[0]

    return make_problem(f, project_x=project_x, project_y=project_y)
