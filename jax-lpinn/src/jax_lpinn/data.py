"""ADE ensemble snapshots and mean/fluctuation split (Eqs 6, 12, 28–29)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_lpinn.model import LPINNConfig
from jax_lpinn.physics import ogata_banks, ogata_banks_derivs


def space_time_mesh(Nx: int, Nt: int, L: float, T: float, exclude_t0: bool = True):
    """Uniform mesh. Paper: t1 = T/(Nt-1), t_Nt = T, x1 = 0, x_Nx = L."""
    x = jnp.linspace(0.0, L, Nx)
    if exclude_t0:
        t = jnp.linspace(T / max(Nt - 1, 1), T, Nt)
    else:
        t = jnp.linspace(0.0, T, Nt)
    xx, tt = jnp.meshgrid(x, t, indexing="ij")
    return x, t, xx, tt


def flatten_mesh(xx, tt):
    return xx.reshape(-1), tt.reshape(-1)


def sample_vd(key, n: int, V_star: float, D_star: float, rel: float = 0.3):
    """Independent Uniform[(1-rel)μ*, (1+rel)μ*] draws of (V, D)."""
    kV, kD = jax.random.split(key)
    V = jax.random.uniform(kV, (n,), minval=(1.0 - rel) * V_star, maxval=(1.0 + rel) * V_star)
    D = jax.random.uniform(kD, (n,), minval=(1.0 - rel) * D_star, maxval=(1.0 + rel) * D_star)
    return V, D


def snapshot_field(x_flat, t_flat, V, D, u0=1.0):
    return ogata_banks(x_flat, t_flat, V, D, u0)


def make_ade_ensemble(
    key,
    n: int,
    Nx: int,
    Nt: int,
    cfg: LPINNConfig,
    exclude_t0: bool = True,
    with_derivs: bool = False,
):
    """Synthetic Ogata–Banks snapshots, shape (n, Nx, Nt)."""
    x, t, xx, tt = space_time_mesh(Nx, Nt, cfg.L, cfg.T, exclude_t0=exclude_t0)
    x_flat, t_flat = flatten_mesh(xx, tt)
    V, D = sample_vd(key, n, cfg.V_star, cfg.D_star)
    shp = (Nx, Nt)

    def one_u(Vi, Di):
        return ogata_banks(x_flat, t_flat, Vi, Di, cfg.u0).reshape(shp)

    U = jax.vmap(one_u)(V, D)
    out = {
        "x": x,
        "t": t,
        "xx": xx,
        "tt": tt,
        "V": V,
        "D": D,
        "U": U,
    }
    if with_derivs:

        def one_d(Vi, Di):
            u, ut, ux, uxx = ogata_banks_derivs(x_flat, t_flat, Vi, Di, cfg.u0)
            return u.reshape(shp), ut.reshape(shp), ux.reshape(shp), uxx.reshape(shp)

        _, Ut, Ux, Uxx = jax.vmap(one_d)(V, D)
        out.update({"Ut": Ut, "Ux": Ux, "Uxx": Uxx})
    return out


def split_mean_fluct(U, Ut=None, Ux=None, Uxx=None):
    """ū = sample mean over the ensemble (Eq 12); u' = u - ū (Eq 6)."""
    U_bar = jnp.mean(U, axis=0)
    U_prime = U - U_bar
    out = {"U_bar": U_bar, "U_prime": U_prime}
    if Ut is not None:
        out["Ut_bar"] = jnp.mean(Ut, axis=0)
        out["Ut_prime"] = Ut - out["Ut_bar"]
    if Ux is not None:
        out["Ux_bar"] = jnp.mean(Ux, axis=0)
        out["Ux_prime"] = Ux - out["Ux_bar"]
    if Uxx is not None:
        out["Uxx_bar"] = jnp.mean(Uxx, axis=0)
        out["Uxx_prime"] = Uxx - out["Uxx_bar"]
    return out


def interior_collocation(Nx: int, Nt: int, L: float, T: float):
    """Interior residual points 0 < x < L, t > 0."""
    x = jnp.linspace(0.0, L, Nx)[1:-1]
    t = jnp.linspace(0.0, T, Nt)[1:]
    xx, tt = jnp.meshgrid(x, t, indexing="ij")
    return xx.reshape(-1), tt.reshape(-1)


def ic_collocation(Nx: int, L: float):
    """Initial-condition points t=0, 0 < x ≤ L (drop the (0,0) discontinuity)."""
    x = jnp.linspace(0.0, L, Nx)[1:]
    t = jnp.zeros_like(x)
    return x, t


def bc_dirichlet_collocation(Nt: int, T: float):
    """Inlet Dirichlet: x=0, t > 0."""
    t = jnp.linspace(0.0, T, Nt)[1:]
    x = jnp.zeros_like(t)
    return x, t


def bc_neumann_collocation(Nt: int, L: float, T: float):
    """Outlet Neumann: x=L, t > 0."""
    t = jnp.linspace(0.0, T, Nt)[1:]
    x = jnp.full_like(t, L)
    return x, t
