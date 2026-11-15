"""PDE residual operators via jax.grad / hessian."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_pirfn.features import cosine_features, product_features

Array = jax.Array


def _as_col(x: Array) -> Array:
    x = jnp.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def apply_1d(x: Array, omega: Array, b: Array, c: Array) -> Array:
    """Evaluate u(x) = phi(x) c for 1D or multi-D concatenated features."""
    phi = cosine_features(_as_col(x), omega, b)
    return phi @ c


def apply_product(
    x: Array,
    t: Array,
    omega_x: Array,
    b_x: Array,
    omega_t: Array,
    b_t: Array,
    c: Array,
) -> Array:
    """Evaluate product-feature expansion u(x, t) = (phi_x ⊗ phi_t) c."""
    phi = product_features(x, t, omega_x, b_x, omega_t, b_t)
    return phi @ c


def _u_scalar_1d(x: Array, omega: Array, b: Array, c: Array) -> Array:
    x = jnp.reshape(x, ())
    return jnp.dot(jnp.cos(omega[:, 0] * x + b), c)


def u_x_1d(x: Array, omega: Array, b: Array, c: Array) -> Array:
    return jax.vmap(lambda z: jax.grad(_u_scalar_1d)(z, omega, b, c))(x)


def u_xx_1d(x: Array, omega: Array, b: Array, c: Array) -> Array:
    def d2(z):
        return jax.grad(jax.grad(_u_scalar_1d))(z, omega, b, c)

    return jax.vmap(d2)(x)


def _u_scalar_nd(x: Array, omega: Array, b: Array, c: Array) -> Array:
    return jnp.dot(jnp.cos(omega @ x + b), c)


def laplacian(xs: Array, omega: Array, b: Array, c: Array) -> Array:
    """Trace Hessian of u at each row of xs, shape (n, d)."""

    def lap_one(x):
        return jnp.trace(jax.hessian(_u_scalar_nd)(x, omega, b, c))

    return jax.vmap(lap_one)(xs)


def _u_product_scalar(
    x: Array,
    t: Array,
    omega_x: Array,
    b_x: Array,
    omega_t: Array,
    b_t: Array,
    c: Array,
) -> Array:
    x = jnp.reshape(x, ())
    t = jnp.reshape(t, ())
    phi_x = jnp.cos(omega_x[:, 0] * x + b_x)
    phi_t = jnp.cos(omega_t[:, 0] * t + b_t)
    return jnp.dot(jnp.outer(phi_x, phi_t).reshape(-1), c)


def product_u_t(
    x: Array,
    t: Array,
    omega_x: Array,
    b_x: Array,
    omega_t: Array,
    b_t: Array,
    c: Array,
) -> Array:
    def dt(xi, ti):
        return jax.grad(_u_product_scalar, argnums=1)(
            xi, ti, omega_x, b_x, omega_t, b_t, c
        )

    return jax.vmap(dt)(x, t)


def product_u_xx(
    x: Array,
    t: Array,
    omega_x: Array,
    b_x: Array,
    omega_t: Array,
    b_t: Array,
    c: Array,
) -> Array:
    def dxx(xi, ti):
        return jax.grad(jax.grad(_u_product_scalar, argnums=0), argnums=0)(
            xi, ti, omega_x, b_x, omega_t, b_t, c
        )

    return jax.vmap(dxx)(x, t)


def poisson_1d_source(x: Array) -> tuple[Array, Array]:
    """Manufactured -u'' = f on [0, 1] with u* = sin(2 pi x)."""
    x = jnp.asarray(x).reshape(-1)
    u = jnp.sin(2.0 * jnp.pi * x)
    f = (2.0 * jnp.pi) ** 2 * u
    return u, f


def helmholtz_1d_source(x: Array, kappa: float = 8.0) -> tuple[Array, Array]:
    """Manufactured -u'' - kappa^2 u = f with u* = sin(2 pi x)."""
    x = jnp.asarray(x).reshape(-1)
    u = jnp.sin(2.0 * jnp.pi * x)
    f = ((2.0 * jnp.pi) ** 2 - kappa ** 2) * u
    return u, f


def heat_1d_exact(x: Array, t: Array) -> Array:
    """u(x, t) = sin(pi x) exp(-pi^2 t)."""
    x = jnp.asarray(x).reshape(-1)
    t = jnp.asarray(t).reshape(-1)
    return jnp.sin(jnp.pi * x) * jnp.exp(-(jnp.pi ** 2) * t)


def poisson_2d_source(xy: Array) -> tuple[Array, Array]:
    """Manufactured -Delta u = f on [0, 1]^2 with u* = sin(pi x) sin(pi y)."""
    xy = jnp.asarray(xy)
    u = jnp.sin(jnp.pi * xy[:, 0]) * jnp.sin(jnp.pi * xy[:, 1])
    f = 2.0 * (jnp.pi ** 2) * u
    return u, f


def poisson_residual_1d(x: Array, omega: Array, b: Array, c: Array, f: Array) -> Array:
    return -u_xx_1d(x, omega, b, c) - f


def helmholtz_residual_1d(
    x: Array, omega: Array, b: Array, c: Array, f: Array, kappa: float = 8.0
) -> Array:
    u = apply_1d(x, omega, b, c)
    return -u_xx_1d(x, omega, b, c) - (kappa ** 2) * u - f


def heat_residual(
    x: Array,
    t: Array,
    omega_x: Array,
    b_x: Array,
    omega_t: Array,
    b_t: Array,
    c: Array,
) -> Array:
    return product_u_t(x, t, omega_x, b_x, omega_t, b_t, c) - product_u_xx(
        x, t, omega_x, b_x, omega_t, b_t, c
    )


def make_poisson_1d_loss(omega, b, x_int, f_int, x_bd, g_bd, lam_l2=1e-6):
    from jax_pirfn.losses import pirfn_loss

    def loss_fn(c):
        r = poisson_residual_1d(x_int, omega, b, c, f_int)
        bc = apply_1d(x_bd, omega, b, c) - g_bd
        return pirfn_loss(r, bc, c, lam_l2=lam_l2)

    return loss_fn


def make_helmholtz_1d_loss(omega, b, x_int, f_int, x_bd, g_bd, kappa=8.0, lam_l2=1e-6):
    from jax_pirfn.losses import pirfn_loss

    def loss_fn(c):
        r = helmholtz_residual_1d(x_int, omega, b, c, f_int, kappa=kappa)
        bc = apply_1d(x_bd, omega, b, c) - g_bd
        return pirfn_loss(r, bc, c, lam_l2=lam_l2)

    return loss_fn


def make_heat_1d_loss(
    omega_x, b_x, omega_t, b_t, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, lam_l2=1e-6
):
    from jax_pirfn.losses import pirfn_loss

    def loss_fn(c):
        r = heat_residual(x_int, t_int, omega_x, b_x, omega_t, b_t, c)
        ic = apply_product(x_ic, t_ic, omega_x, b_x, omega_t, b_t, c) - u_ic
        bc = apply_product(x_bc, t_bc, omega_x, b_x, omega_t, b_t, c) - u_bc
        return pirfn_loss(r, bc, c, ic_residual=ic, lam_l2=lam_l2)

    return loss_fn
