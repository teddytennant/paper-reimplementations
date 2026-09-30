"""Shared mass-compression-transport step and known-law supervision.

State is a positive density on a periodic 1D grid. Factors satisfy rho = M * I,
with I the compression and M the mass per reference volume. Cell centers are
x_i = i * dx on [0, N * dx).

The paper does not claim exact discrete mass conservation of the reconstructed
density. Periodic fluxes cancel in the sum of I, and the logarithmic reaction
update preserves M > 0 in exact arithmetic, but the product rho = M * I is not
exactly conserved by the split step. Positivity of rho is not guaranteed.
Pass clip=True to floor the reconstructed density at 1e-8. That floor is not
part of the paper scheme, and it does not rewrite I or M.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

DENSITY_FLOOR = 1e-8


def _require_positive_number(value, name):
    """Raise ValueError if a concrete scalar is not strictly positive.

    Traced values are left to the caller. A Python or NumPy/JAX scalar is
    checked immediately so dt <= 0 and dx <= 0 fail outside JIT.
    """
    if isinstance(value, jax.core.Tracer):
        return
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive scalar") from exc
    if not (number > 0.0):
        raise ValueError(f"{name} must be positive, got {number}")


def _require_positive_density(rho):
    """Raise ValueError if a concrete density has a nonpositive entry."""
    if isinstance(rho, jax.core.Tracer):
        return
    if bool(jnp.any(jnp.asarray(rho) <= 0)):
        raise ValueError("rho must be positive")


def periodic_central_diff(field, spacing, axis=-1):
    """Periodic central difference (q[i+1] - q[i-1]) / (2 * spacing)."""
    field = jnp.asarray(field)
    spacing = jnp.asarray(spacing, dtype=field.dtype)
    forward = jnp.roll(field, -1, axis=axis)
    backward = jnp.roll(field, 1, axis=axis)
    return (forward - backward) / (2.0 * spacing)


def finite_volume_Lv(I, v, dx):
    """Unsplit upwind finite-volume operator Lv of -d/dx (I v).

    Face velocity between cell i and i+1 is the arithmetic mean of v[i] and
    v[i+1]. The face flux is the upwind value of I times that face velocity.
    (Lv I)[i] = -(flux[i] - flux[i-1]) / dx.
    """
    _require_positive_number(dx, "dx")
    I = jnp.asarray(I)
    v = jnp.asarray(v)
    dx_arr = jnp.asarray(dx, dtype=I.dtype)
    v_face = 0.5 * (v + jnp.roll(v, -1))
    i_up = jnp.where(v_face >= 0.0, I, jnp.roll(I, -1))
    flux = v_face * i_up
    return -(flux - jnp.roll(flux, 1)) / dx_arr


def ssprk2_compression(I, v, dt, dx):
    """SSPRK2 compression update with velocity frozen (Equation 20).

    I1 = I + dt * Lv(I)
    I_next = 0.5 * I + 0.5 * I1 + 0.5 * dt * Lv(I1)
    """
    _require_positive_number(dt, "dt")
    _require_positive_number(dx, "dx")
    I = jnp.asarray(I)
    v = jnp.asarray(v)

    def Lv(factor):
        return finite_volume_Lv(factor, v, dx)

    i1 = I + dt * Lv(I)
    return 0.5 * I + 0.5 * i1 + 0.5 * dt * Lv(i1)


def periodic_linear_interp(values, x_query, dx):
    """Periodic linear interpolation of a grid field at arbitrary stations.

    values live at x_i = i * dx. Queries are wrapped into [0, N * dx).
    """
    values = jnp.asarray(values)
    n = values.shape[0]
    dx_arr = jnp.asarray(dx, dtype=values.dtype)
    length = n * dx_arr
    wrapped = jnp.mod(jnp.asarray(x_query, dtype=values.dtype), length)
    s = wrapped / dx_arr
    i0 = jnp.floor(s).astype(jnp.int32)
    alpha = s - i0.astype(values.dtype)
    i0 = jnp.mod(i0, n)
    i1 = jnp.mod(i0 + 1, n)
    return (1.0 - alpha) * values[i0] + alpha * values[i1]


def semilag_mass(M, v, dt, dx):
    """Midpoint semi-Lagrangian update of M (Equation 21).

    xmid = x - 0.5 * dt * v(x)
    xdep = x - dt * interpolate(v, xmid)
    M_next(x) = interpolate(M, xdep)

    v is a frozen grid field. This does not reevaluate a constitutive map.
    """
    _require_positive_number(dt, "dt")
    _require_positive_number(dx, "dx")
    M = jnp.asarray(M)
    v = jnp.asarray(v)
    dtype = M.dtype
    dt_arr = jnp.asarray(dt, dtype=dtype)
    dx_arr = jnp.asarray(dx, dtype=dtype)
    x = jnp.arange(M.shape[0], dtype=dtype) * dx_arr
    xmid = x - 0.5 * dt_arr * v
    v_mid = periodic_linear_interp(v, xmid, dx_arr)
    xdep = x - dt_arr * v_mid
    return periodic_linear_interp(M, xdep, dx_arr)


def reaction_step(M, I, rate_fn, h):
    """Explicit midpoint reaction update of M with I held fixed (Equation 19).

    rate_fn(M, I) returns the relative rate r, pointwise. None means the
    system is nonreactive and the step returns M unchanged.

    With m = log M:
        k1 = rate_fn(exp(m), I)
        k2 = rate_fn(exp(m + (h/2) * k1), I)
        M_next = exp(m + h * k2)
    The density argument of the paper's rate map R is I * M at each stage,
    so a rate_fn that depends on density should use the product M * I.
    """
    if rate_fn is None:
        return M
    M = jnp.asarray(M)
    I = jnp.asarray(I)
    m = jnp.log(M)
    k1 = rate_fn(jnp.exp(m), I)
    k2 = rate_fn(jnp.exp(m + (0.5 * h) * k1), I)
    return jnp.exp(m + h * k2)


def mct_step(rho, I, M, velocity_fn, dt, dx, rate_fn=None, clip=False):
    """One nonreactive or reactive MCT step.

    v = velocity_fn(rho) is frozen for the whole step. The split is a reaction
    half-step, SSPRK2 compression, semi-Lagrangian mass transport, and a
    second reaction half-step, then rho = M * I. If I or M is None, start from
    I = 1 and M = rho.

    clip=False by default. clip=True floors only the reconstructed density at
    1e-8 and does not rewrite the factors. The paper does not claim exact
    discrete mass conservation of rho.
    """
    _require_positive_number(dt, "dt")
    _require_positive_number(dx, "dx")
    rho = jnp.asarray(rho)
    if rho.ndim != 1:
        raise ValueError("rho must be a 1D periodic grid")
    if I is None or M is None:
        I = jnp.ones_like(rho)
        M = rho
    else:
        I = jnp.asarray(I)
        M = jnp.asarray(M)
    v = jnp.asarray(velocity_fn(rho))
    h = 0.5 * dt
    M = reaction_step(M, I, rate_fn, h)
    I = ssprk2_compression(I, v, dt, dx)
    M = semilag_mass(M, v, dt, dx)
    M = reaction_step(M, I, rate_fn, h)
    rho_next = M * I
    if clip:
        rho_next = jnp.maximum(rho_next, DENSITY_FLOOR)
    return rho_next, I, M


def rollout(rho0, velocity_fn, dt, dx, steps, rate_fn=None, t_reinit=None, clip=False):
    """Roll the split step. Returns density history of shape (steps + 1, N).

    t_reinit=None never resets the factors. Otherwise, after a step, if the
    time since the last reset is at least t_reinit, set I = 1 and M = rho and
    clear the accumulator. Velocity is reevaluated only at the start of each
    step, from the density passed into that step.
    """
    _require_positive_number(dt, "dt")
    _require_positive_number(dx, "dx")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    rho = jnp.asarray(rho0)
    if rho.ndim != 1:
        raise ValueError("rho must be a 1D periodic grid")
    I = jnp.ones_like(rho)
    M = rho
    frames = [rho]
    tau = 0.0
    dt_value = float(dt)
    for _ in range(int(steps)):
        rho, I, M = mct_step(
            rho, I, M, velocity_fn, dt, dx, rate_fn=rate_fn, clip=clip
        )
        tau = tau + dt_value
        if t_reinit is not None and tau >= float(t_reinit):
            I = jnp.ones_like(rho)
            M = rho
            tau = 0.0
        frames.append(rho)
    return jnp.stack(frames, axis=0)


def known_law_targets(rho, D, dx):
    """Linear-diffusion constitutive targets on a periodic grid.

    xi = 1 / rho, f = -D * central_diff(rho). Raises ValueError if dx <= 0 or
    if any concrete rho entry is nonpositive (log rho is then undefined).
    Accepts a single field (N,) or a batch (B, N). The derivative is along the
    last axis.
    """
    _require_positive_number(dx, "dx")
    rho = jnp.asarray(rho)
    _require_positive_density(rho)
    xi = 1.0 / rho
    f = -D * periodic_central_diff(rho, dx, axis=-1)
    return xi, f


def linear_diffusion_velocity(rho, D, dx):
    """Discrete transport velocity u = xi * f = -D * central_diff(rho) / rho.

    In the continuum this is -D * d(log rho) / dx. The grid form uses the
    product of the known-law factors so the constitutive pair stays consistent.
    """
    xi, f = known_law_targets(rho, D, dx)
    return xi * f


def _stencils(rho, dx):
    """Local stencil (rho[i-1], rho[i], rho[i+1], central slope) on the last axis."""
    left = jnp.roll(rho, 1, axis=-1)
    right = jnp.roll(rho, -1, axis=-1)
    slope = (right - left) / (2.0 * jnp.asarray(dx, dtype=rho.dtype))
    return jnp.stack([left, rho, right, slope], axis=-1)


def _predict_factors(model, rho, dx):
    stencils = _stencils(rho, dx)
    leading = stencils.shape[:-1]
    flat = stencils.reshape((-1, 4))
    log_xi, f_raw = jax.vmap(model)(flat)
    return log_xi.reshape(leading), f_raw.reshape(leading)


class LocalConstitutive(eqx.Module):
    """Tiny local map from a 4-point stencil to (log_xi, f_raw).

    The stencil is (rho[i-1], rho[i], rho[i+1], central-difference slope).
    The module does not see the rest of the field. Known-law loss compares
    the returned log_xi and f_raw with log(1/rho) and -D * d(rho)/dx.
    A velocity built from the same outputs is softplus(log_xi) * f_raw.
    """

    mlp: eqx.nn.MLP
    skip: eqx.nn.Linear

    def __init__(self, key, width=32, depth=2):
        k_mlp, k_skip = jax.random.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=4,
            out_size=2,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=k_mlp,
        )
        self.skip = eqx.nn.Linear(4, 2, key=k_skip)

    def __call__(self, stencil):
        stencil = jnp.asarray(stencil)
        out = self.mlp(stencil) + self.skip(stencil)
        return out[0], out[1]


def known_law_loss(model, rho, D, dx):
    """Mean squared error on log mobility and force, not on their product.

    L = mean( (log_xi_hat - log xi_true)^2 + (f_hat - f_true)^2 ).
    No time integration. rho may be one field (N,) or a batch (B, N).
    """
    _require_positive_number(dx, "dx")
    rho = jnp.asarray(rho)
    xi_true, f_true = known_law_targets(rho, D, dx)
    log_xi_hat, f_hat = _predict_factors(model, rho, dx)
    residual = (log_xi_hat - jnp.log(xi_true)) ** 2 + (f_hat - f_true) ** 2
    return jnp.mean(residual)


def velocity_loss(model, rho, D, dx):
    """Mean squared error of softplus(log_xi) * f_raw against xi_true * f_true.

    A rescaling xi' = lambda * xi, f' = f / lambda leaves this loss unchanged
    and changes known_law_loss. Velocity data alone do not identify the factors.
    """
    _require_positive_number(dx, "dx")
    rho = jnp.asarray(rho)
    u_true = linear_diffusion_velocity(rho, D, dx)
    log_xi, f_raw = _predict_factors(model, rho, dx)
    u_hat = jax.nn.softplus(log_xi) * f_raw
    return jnp.mean((u_hat - u_true) ** 2)


def curl_penalty(f):
    """Mean squared curl of a driving force.

    A 1D force has identically zero curl, so this returns 0. For a 2D field
    of shape (2, H, W), with f[0] = f_x and f[1] = f_y, axis -2 = y and
    axis -1 = x, the penalty is

        mean( (d f_y / d x - d f_x / d y)^2 )

    using periodic central differences with grid spacing 1 (one cell). The
    same operator is periodic_central_diff(..., spacing=1). This is the 2D
    finite-difference check used under velocity-data supervision. It is not
    applied during known-law training.
    """
    f = jnp.asarray(f)
    if f.ndim < 3 or f.shape[0] != 2:
        return jnp.zeros((), dtype=jnp.result_type(f, jnp.float32))
    fx = f[0]
    fy = f[1]
    dfy_dx = periodic_central_diff(fy, spacing=1.0, axis=-1)
    dfx_dy = periodic_central_diff(fx, spacing=1.0, axis=-2)
    return jnp.mean((dfy_dx - dfx_dy) ** 2)
