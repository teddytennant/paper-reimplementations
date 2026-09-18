"""1D advection–diffusion operator and Ogata–Banks snapshots (Eqs 23–27)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import erfc, erfcx

from jax_lpinn.model import features, mean_value


def ogata_banks(x, t, V, D, u0=1.0):
    """Closed-form ADE solution on the half-line (Eq 27).

    u(x,t) = (u0/2) [erfc((x-Vt)/(2√(Dt))) + exp(Vx/D) erfc((x+Vt)/(2√(Dt)))]

    The second term is evaluated as exp(-(x-Vt)^2/(4Dt)) erfcx((x+Vt)/(2√(Dt)))
    so that exp(Vx/D) does not overflow. At t=0 the IC u=0 is returned.
    """
    x = jnp.asarray(x)
    t = jnp.asarray(t)
    V = jnp.asarray(V)
    D = jnp.asarray(D)
    t_safe = jnp.maximum(t, 1e-30)
    inv_s = 1.0 / (2.0 * jnp.sqrt(D * t_safe))
    z1 = (x - V * t_safe) * inv_s
    z2 = (x + V * t_safe) * inv_s
    term1 = erfc(z1)
    log_scale = -((x - V * t_safe) ** 2) / (4.0 * D * t_safe)
    term2 = jnp.exp(log_scale) * erfcx(z2)
    u = 0.5 * u0 * (term1 + term2)
    return jnp.where(t > 0, u, jnp.zeros_like(u))


def scalar_derivs(u_fn, x, t):
    """Autodiff u, u_t, u_x, u_xx for a scalar field u_fn(x, t)."""

    def one(xx, tt):
        def f(x_, t_):
            return u_fn(x_, t_)

        u, ut = jax.value_and_grad(f, argnums=1)(xx, tt)
        ux = jax.grad(f, argnums=0)(xx, tt)
        uxx = jax.grad(lambda x_: jax.grad(f, argnums=0)(x_, tt))(xx)
        return u, ut, ux, uxx

    return jax.vmap(one)(x, t)


def vector_derivs(v_fn, x, t):
    """Autodiff v, v_t, v_x, v_xx for a vector field v_fn(x, t) -> (M,)."""

    def one(xx, tt):
        v = v_fn(xx, tt)
        vt = jax.jacfwd(lambda t_: v_fn(xx, t_))(tt)
        vx = jax.jacfwd(lambda x_: v_fn(x_, tt))(xx)
        vxx = jax.jacfwd(lambda x_: jax.jacfwd(lambda x2: v_fn(x2, tt))(x_))(xx)
        return v, vt, vx, vxx

    return jax.vmap(one)(x, t)


def ade_operator(ut, ux, uxx, V, D):
    """L(u; V, D) = u_t + V u_x - D u_xx (Eq 23)."""
    return ut + V * ux - D * uxx


def mean_derivs(mean_params, x, t, *, L: float, T: float):
    def u_fn(xx, tt):
        return mean_value(mean_params, xx, tt, L=L, T=T)

    return scalar_derivs(u_fn, x, t)


def feature_derivs(hidden_params, x, t, *, L: float, T: float):
    def v_fn(xx, tt):
        return features(hidden_params, xx, tt, L=L, T=T)

    return vector_derivs(v_fn, x, t)


def ogata_banks_derivs(x, t, V, D, u0=1.0):
    def u_fn(xx, tt):
        return ogata_banks(xx, tt, V, D, u0)

    return scalar_derivs(u_fn, x, t)
