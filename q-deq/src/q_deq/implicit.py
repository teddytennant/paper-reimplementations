"""Implicit differentiation through the DEQ root z* = F(z*, x).

Solves (I - ∂F/∂z)^T w = cotangent with a dense linear solve, not a
Neumann one-step approximation and not unrolled solver history.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def jacobian_wrt_z(F, z, x):
    return jax.jacobian(lambda zz: F(zz, x))(z)


def implicit_adjoint(F, z_star, x, cotangent):
    """w = (I - ∂F/∂z)^{-T} cotangent, evaluated at (z_star, x)."""
    J = jacobian_wrt_z(F, z_star, x)
    d = z_star.shape[-1]
    I = jnp.eye(d, dtype=z_star.dtype)
    return jnp.linalg.solve((I - J).T, cotangent)


def make_deq_rewire(F_apply):
    """Forward: W8A8 (or fp) re-forward F(params, z*, x).

    Backward: IFT VJP, w = (I - ∂F/∂z)^{-T} g, then VJP of F w.r.t. params/x.
    Solver history is not differentiated (z* is treated as given).
    """

    @jax.custom_vjp
    def rewire(params, z_star, x):
        return F_apply(params, z_star, x)

    def fwd(params, z_star, x):
        z_out = F_apply(params, z_star, x)
        return z_out, (params, z_star, x)

    def bwd(res, g):
        params, z_star, x = res

        w = implicit_adjoint(lambda z, xx: F_apply(params, z, xx), z_star, x, g)
        _, vjp = jax.vjp(lambda p, xx: F_apply(p, z_star, xx), params, x)
        dp, dx = vjp(w)
        return dp, jnp.zeros_like(z_star), dx

    rewire.defvjp(fwd, bwd)
    return rewire
