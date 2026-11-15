"""Algorithm 1: anchored dyadic-minibatch SGD with activity-block β updates."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, random

from dls_pp.intensity import intensity, log_intensity, pairwise_squared_distances
from dls_pp.likelihood import Events, dyad_indices, poisson_loglik, quadrature_grid
from dls_pp.penalties import omega_matrices, roughness_penalty
from dls_pp.splines import (
    SPLINE_DEGREE,
    basis_matrix,
    evaluate_trajectories,
    greville_abscissae,
    make_basis,
)


def apply_anchor(c_x: Array, c_y: Array) -> tuple[Array, Array]:
    """Hold c_1^x = c_1^y = 0 and c_2^y = 0 (eq 2.12, set C0)."""
    c_x = c_x.at[0].set(0.0)
    c_y = c_y.at[0].set(0.0)
    c_y = c_y.at[1].set(0.0)
    return c_x, c_y


def project_positions_c0(z: Array) -> Array:
    """Translate node 1 to the origin and rotate so node 2 has y=0.

    For trajectories of shape (n, M, 2) the rotation is taken from t of the
    first time index, then applied at every t.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    z = z - z[0]
    if z.ndim == 2:
        xy = z[1]
    else:
        xy = z[1, 0]
    theta = jnp.arctan2(xy[1], xy[0])
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    rot = jnp.array([[c, s], [-s, c]], dtype=z.dtype)
    return z @ rot.T


def project_coefficients_c0(c_x: Array, c_y: Array) -> tuple[Array, Array]:
    """Translate in coefficient space, rotate from z_2(0), then enforce C0."""
    c_x = c_x - c_x[0]
    c_y = c_y - c_y[0]
    theta = jnp.arctan2(c_y[1, 0], c_x[1, 0])
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    nx = c * c_x + s * c_y
    ny = -s * c_x + c * c_y
    return apply_anchor(nx, ny)


def constant_coefficients(z: Array, n_coeff: int) -> tuple[Array, Array]:
    """Spline coefficients of static positions (clamped cubics reproduce constants)."""
    c_x = jnp.broadcast_to(z[:, 0:1], (z.shape[0], n_coeff))
    c_y = jnp.broadcast_to(z[:, 1:2], (z.shape[0], n_coeff))
    return jnp.asarray(c_x, dtype=jnp.float64), jnp.asarray(c_y, dtype=jnp.float64)


def linear_coefficients(intercept: Array, slope: Array, knots: Array) -> tuple[Array, Array]:
    """Coefficients of z_i(t) = intercept_i + slope_i * t (affine reproduction)."""
    grev = greville_abscissae(knots)
    c_x = intercept[:, 0:1] + slope[:, 0:1] * grev[None, :]
    c_y = intercept[:, 1:2] + slope[:, 1:2] * grev[None, :]
    return c_x, c_y


def _event_counts(events: Events, n: int) -> Array:
    counts = jnp.zeros((n, n), dtype=jnp.float64)
    if events.times.shape[0] == 0:
        return counts
    counts = counts.at[events.src, events.dst].add(1.0)
    return counts + counts.T


def _mu_matrix(
    beta: Array,
    c_x: Array,
    c_y: Array,
    phi_q: Array,
    quad_w: Array,
    model: str,
) -> Array:
    n = beta.shape[0]
    src, dst = dyad_indices(n)
    zx = c_x @ phi_q.T
    zy = c_y @ phi_q.T
    z = jnp.stack([zx, zy], axis=-1)
    log_lam = log_intensity(
        beta[src][:, None],
        beta[dst][:, None],
        z[src],
        z[dst],
        model=model,
    )
    mu_pairs = jnp.sum(jnp.exp(log_lam) * quad_w[None, :], axis=-1)
    mu = jnp.zeros((n, n), dtype=jnp.float64)
    mu = mu.at[src, dst].set(mu_pairs)
    return mu + mu.T


def activity_block_solve(
    beta: Array,
    c_x: Array,
    c_y: Array,
    events: Events,
    phi_q: Array,
    quad_w: Array,
    model: str = "full",
    n_iter: int = 8,
    ridge: float = 1e-8,
) -> Array:
    """Newton solve for β given trajectories (activity block of Alg. 1)."""
    n = beta.shape[0]
    n_mat = _event_counts(events, n)

    def body(_k, beta):
        mu = _mu_matrix(beta, c_x, c_y, phi_q, quad_w, model)
        g = jnp.sum(n_mat - mu, axis=1)
        h = -mu
        h = h.at[jnp.diag_indices(n)].set(-jnp.sum(mu, axis=1))
        h = h - ridge * jnp.eye(n, dtype=h.dtype)
        delta = jnp.linalg.solve(h, -g)
        return beta + delta

    return jax.lax.fori_loop(0, n_iter, body, beta)


def _ll_from_phi(
    beta: Array,
    c_x: Array,
    c_y: Array,
    events: Events,
    phi_e: Array,
    phi_q: Array,
    quad_w: Array,
    model: str,
    src: Array | None,
    dst: Array | None,
) -> Array:
    n = beta.shape[0]
    if events.times.shape[0] == 0:
        event_term = jnp.array(0.0, dtype=jnp.float64)
    else:
        if src is None:
            mask = jnp.ones(events.times.shape[0], dtype=jnp.float64)
        else:
            event_ids = events.src.astype(jnp.int32) * n + events.dst.astype(jnp.int32)
            batch_ids = src.astype(jnp.int32) * n + dst.astype(jnp.int32)
            mask = jnp.isin(event_ids, batch_ids).astype(jnp.float64)
        if model == "beta_only":
            log_lam_e = beta[events.src] + beta[events.dst]
        else:
            z_src = jnp.stack(
                [
                    jnp.sum(c_x[events.src] * phi_e, axis=-1),
                    jnp.sum(c_y[events.src] * phi_e, axis=-1),
                ],
                axis=-1,
            )
            z_dst = jnp.stack(
                [
                    jnp.sum(c_x[events.dst] * phi_e, axis=-1),
                    jnp.sum(c_y[events.dst] * phi_e, axis=-1),
                ],
                axis=-1,
            )
            log_lam_e = log_intensity(
                beta[events.src], beta[events.dst], z_src, z_dst, model=model
            )
        event_term = jnp.sum(mask * log_lam_e)
    integ_src, integ_dst = (src, dst) if src is not None else dyad_indices(n)
    zx = c_x @ phi_q.T
    zy = c_y @ phi_q.T
    z = jnp.stack([zx, zy], axis=-1)
    log_lam_q = log_intensity(
        beta[integ_src][:, None],
        beta[integ_dst][:, None],
        z[integ_src],
        z[integ_dst],
        model=model,
    )
    integ = jnp.sum(jnp.exp(log_lam_q) * quad_w[None, :])
    return event_term - integ


def minibatch_objective(
    c_x: Array,
    c_y: Array,
    beta: Array,
    events: Events,
    phi_e: Array,
    phi_q: Array,
    quad_w: Array,
    src: Array,
    dst: Array,
    omega1: Array,
    omega2: Array,
    tau: float,
    rho1: float,
    rho2: float,
    n_dyads: int,
    model: str,
) -> Array:
    """Scaled minibatch J(c, β) = ℓ − P (eq 2.11), likelihood over the batch."""
    ll = _ll_from_phi(beta, c_x, c_y, events, phi_e, phi_q, quad_w, model, src, dst)
    scale = n_dyads / jnp.maximum(src.shape[0], 1)
    pen = roughness_penalty(c_x, c_y, omega1, omega2, tau, rho1, rho2)
    return scale * ll - pen


@dataclass
class FitResult:
    beta: Array
    c_x: Array
    c_y: Array
    loglik: Array
    history: Array


def fit_dls_pp(
    events: Events,
    n_coeff: int = 6,
    n_steps: int = 200,
    batch_size: int = 8,
    lr: float = 0.05,
    tau: float = 1.0,
    rho1: float = 0.1,
    rho2: float = 0.01,
    n_quad: int = 24,
    model: str = "full",
    seed: int = 0,
    c_x: Array | None = None,
    c_y: Array | None = None,
    beta: Array | None = None,
    beta_every: int = 5,
) -> FitResult:
    """Estimate (c, β) by Alg. 1: trajectory SGD + activity-block β updates."""
    n = events.n_nodes
    t_max = events.t_max
    basis = make_basis(t_max, n_coeff)
    knots = basis.knots
    quad_t, quad_w = quadrature_grid(t_max, n_quad)
    phi_q = basis_matrix(quad_t, knots)
    if events.times.shape[0] > 0:
        phi_e = basis_matrix(events.times, knots)
    else:
        phi_e = jnp.zeros((0, n_coeff), dtype=jnp.float64)
    omega1, omega2 = omega_matrices(knots, t_max, n_quad=max(n_quad, 48))
    key = random.PRNGKey(seed)
    if c_x is None or c_y is None:
        key, k0 = random.split(key)
        noise = 0.3 * random.normal(k0, (n, n_coeff, 2), dtype=jnp.float64)
        c_x = noise[:, :, 0]
        c_y = noise[:, :, 1]
    c_x, c_y = apply_anchor(c_x, c_y)
    if model == "homogeneous":
        z0 = jnp.stack([jnp.mean(c_x, axis=1), jnp.mean(c_y, axis=1)], axis=-1)
        c_x, c_y = apply_anchor(*constant_coefficients(z0, n_coeff))
    if beta is None:
        beta = jnp.zeros(n, dtype=jnp.float64)
    beta = activity_block_solve(beta, c_x, c_y, events, phi_q, quad_w, model=model)
    src_all, dst_all = dyad_indices(n)
    n_dyads = int(src_all.shape[0])
    batch_size = min(int(batch_size), n_dyads)

    def step(carry, _):
        c_x, c_y, beta, key = carry
        key, kb = random.split(key)
        perm = random.permutation(kb, n_dyads)[:batch_size]
        src = src_all[perm]
        dst = dst_all[perm]

        def loss(cx, cy):
            return -minibatch_objective(
                cx,
                cy,
                beta,
                events,
                phi_e,
                phi_q,
                quad_w,
                src,
                dst,
                omega1,
                omega2,
                tau,
                rho1,
                rho2,
                n_dyads,
                model,
            )

        gcx, gcy = jax.grad(loss, argnums=(0, 1))(c_x, c_y)
        c_x = c_x - lr * gcx
        c_y = c_y - lr * gcy
        if model == "homogeneous":
            z0 = jnp.stack([jnp.mean(c_x, axis=1), jnp.mean(c_y, axis=1)], axis=-1)
            c_x, c_y = constant_coefficients(z0, n_coeff)
        c_x, c_y = apply_anchor(c_x, c_y)
        return (c_x, c_y, beta, key), jnp.array(0.0, dtype=jnp.float64)

    history = []
    ll0 = poisson_loglik(beta, c_x, c_y, events, knots, quad_t, quad_w, model=model)
    history.append(ll0)
    carry = (c_x, c_y, beta, key)
    # Python loop so we can refresh β every beta_every steps without tracing issues.
    for s in range(n_steps):
        carry, _ = step(carry, s)
        c_x, c_y, beta, key = carry
        if (s + 1) % beta_every == 0 or s == n_steps - 1:
            beta = activity_block_solve(beta, c_x, c_y, events, phi_q, quad_w, model=model)
            carry = (c_x, c_y, beta, key)
    ll1 = poisson_loglik(beta, c_x, c_y, events, knots, quad_t, quad_w, model=model)
    history.append(ll1)
    return FitResult(
        beta=beta,
        c_x=c_x,
        c_y=c_y,
        loglik=ll1,
        history=jnp.stack(history),
    )


def mean_pairwise_intensity(
    beta: Array,
    c_x: Array,
    c_y: Array,
    times: Array,
    knots: Array,
    model: str = "full",
) -> Array:
    """Time-averaged λ_ij on a grid, shape (n, n)."""
    z = evaluate_trajectories(c_x, c_y, times, knots)
    bi = beta[:, None, None]
    bj = beta[None, :, None]
    if model == "beta_only":
        log_lam = bi + bj
        log_lam = jnp.broadcast_to(log_lam, (beta.shape[0], beta.shape[0], times.shape[0]))
    else:
        d2 = pairwise_squared_distances(z)
        log_lam = bi + bj - d2
    return jnp.mean(jnp.exp(log_lam), axis=-1)


def simulate_network(
    key: Array,
    beta: Array,
    c_x: Array,
    c_y: Array,
    knots: Array,
    t_max: float,
    model: str = "full",
) -> Events:
    """Inhomogeneous Poisson events on undirected dyads via thinning."""
    n = beta.shape[0]
    src_all, dst_all = dyad_indices(n)
    src_list = []
    dst_list = []
    time_list = []
    for a, b in zip(src_all.tolist(), dst_all.tolist()):
        key, k_dyad = random.split(key)
        lam_max = float(jnp.exp(beta[a] + beta[b]))
        key, k1, k2, k3 = random.split(k_dyad, 4)
        n_prop = int(random.poisson(k1, lam_max * t_max))
        if n_prop == 0:
            continue
        times = random.uniform(k2, (n_prop,), minval=0.0, maxval=t_max, dtype=jnp.float64)
        phi = basis_matrix(times, knots)
        if model == "beta_only":
            lam = jnp.full((n_prop,), jnp.exp(beta[a] + beta[b]))
        else:
            z_a = jnp.stack([phi @ c_x[a], phi @ c_y[a]], axis=-1)
            z_b = jnp.stack([phi @ c_x[b], phi @ c_y[b]], axis=-1)
            lam = intensity(beta[a], beta[b], z_a, z_b, model="full")
        keep = random.uniform(k3, (n_prop,), dtype=jnp.float64) < (lam / lam_max)
        kept = times[keep]
        for t in kept.tolist():
            src_list.append(int(a))
            dst_list.append(int(b))
            time_list.append(float(t))
    if not time_list:
        return Events(
            src=jnp.zeros((0,), dtype=jnp.int32),
            dst=jnp.zeros((0,), dtype=jnp.int32),
            times=jnp.zeros((0,), dtype=jnp.float64),
            n_nodes=n,
            t_max=float(t_max),
        )
    order = jnp.argsort(jnp.array(time_list, dtype=jnp.float64))
    return Events(
        src=jnp.array(src_list, dtype=jnp.int32)[order],
        dst=jnp.array(dst_list, dtype=jnp.int32)[order],
        times=jnp.array(time_list, dtype=jnp.float64)[order],
        n_nodes=n,
        t_max=float(t_max),
    )
