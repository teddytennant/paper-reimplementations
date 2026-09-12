"""Poisson log-likelihood of the dynamic latent-space model (eq 2.4)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from dls_pp.intensity import log_intensity
from dls_pp.penalties import gauss_legendre
from dls_pp.splines import basis_matrix, evaluate_trajectories, squared_distance_quadratic


@dataclass(frozen=True)
class Events:
    """Undirected dyadic event times on n nodes over [0, T]."""

    src: Array
    dst: Array
    times: Array
    n_nodes: int
    t_max: float

    @property
    def n_events(self) -> int:
        return int(self.times.shape[0])


def dyad_indices(n: int) -> tuple[Array, Array]:
    """Pairs (i, j) with i < j, the undirected dyad set D."""
    i, j = jnp.triu_indices(n, k=1)
    return i.astype(jnp.int32), j.astype(jnp.int32)


def quadrature_grid(t_max: float, n_quad: int = 32) -> tuple[Array, Array]:
    """Gauss–Legendre nodes and weights on [0, T] for ∫ λ_ij(t) dt."""
    return gauss_legendre(n_quad, 0.0, float(t_max))


def log_intensity_from_coeffs(
    beta_i: Array,
    beta_j: Array,
    c_i_x: Array,
    c_j_x: Array,
    c_i_y: Array,
    c_j_y: Array,
    phi: Array,
    model: str = "full",
) -> Array:
    """log λ_ij(t) from spline coefficients and a single φ(t)."""
    if model == "beta_only":
        return beta_i + beta_j
    d2 = squared_distance_quadratic(c_i_x, c_j_x, c_i_y, c_j_y, phi)
    return beta_i + beta_j - d2


def event_log_intensity(
    beta: Array,
    c_x: Array,
    c_y: Array,
    events: Events,
    knots: Array,
    model: str = "full",
    dyad_src: Array | None = None,
    dyad_dst: Array | None = None,
) -> Array:
    """∑_{t ∈ H_ij} log λ_ij(t), optionally restricted to a dyad subset."""
    if events.times.shape[0] == 0:
        return jnp.array(0.0, dtype=jnp.float64)
    n = events.n_nodes
    mask = jnp.ones(events.times.shape[0], dtype=bool)
    if dyad_src is not None:
        event_ids = events.src.astype(jnp.int32) * n + events.dst.astype(jnp.int32)
        batch_ids = dyad_src.astype(jnp.int32) * n + dyad_dst.astype(jnp.int32)
        mask = jnp.isin(event_ids, batch_ids)
    phi = basis_matrix(events.times, knots)
    if model == "beta_only":
        log_lam = beta[events.src] + beta[events.dst]
    else:
        z_src = jnp.stack(
            [jnp.sum(c_x[events.src] * phi, axis=-1), jnp.sum(c_y[events.src] * phi, axis=-1)],
            axis=-1,
        )
        z_dst = jnp.stack(
            [jnp.sum(c_x[events.dst] * phi, axis=-1), jnp.sum(c_y[events.dst] * phi, axis=-1)],
            axis=-1,
        )
        log_lam = log_intensity(beta[events.src], beta[events.dst], z_src, z_dst, model=model)
    return jnp.sum(jnp.where(mask, log_lam, 0.0))


def integrated_intensity(
    beta: Array,
    c_x: Array,
    c_y: Array,
    knots: Array,
    quad_t: Array,
    quad_w: Array,
    model: str = "full",
    src: Array | None = None,
    dst: Array | None = None,
) -> Array:
    """∑_{(i,j)∈D} ∫_0^T λ_ij(t) dt, optionally restricted to a dyad subset."""
    n = beta.shape[0]
    if src is None or dst is None:
        src, dst = dyad_indices(n)
    z = evaluate_trajectories(c_x, c_y, quad_t, knots)
    log_lam = log_intensity(
        beta[src][:, None],
        beta[dst][:, None],
        z[src],
        z[dst],
        model=model,
    )
    return jnp.sum(jnp.exp(log_lam) * quad_w[None, :])


def poisson_loglik(
    beta: Array,
    c_x: Array,
    c_y: Array,
    events: Events,
    knots: Array,
    quad_t: Array,
    quad_w: Array,
    model: str = "full",
    src: Array | None = None,
    dst: Array | None = None,
) -> Array:
    """ℓ(Z, β) = ∑_D [ ∑_{t∈H_ij} log λ_ij(t) − ∫ λ_ij(t) dt ] (eq 2.4)."""
    event_term = event_log_intensity(
        beta, c_x, c_y, events, knots, model=model, dyad_src=src, dyad_dst=dst
    )
    integ = integrated_intensity(
        beta, c_x, c_y, knots, quad_t, quad_w, model=model, src=src, dst=dst
    )
    return event_term - integ


def homogeneous_loglik(
    beta: Array,
    z: Array,
    events: Events,
    t_max: float,
    model: str = "homogeneous",
) -> Array:
    """Log-likelihood when z is constant in t (homogeneous nested model)."""
    kind = "beta_only" if model == "beta_only" else "full"
    if events.times.shape[0] == 0:
        event_term = jnp.array(0.0, dtype=jnp.float64)
    else:
        event_term = jnp.sum(
            log_intensity(beta[events.src], beta[events.dst], z[events.src], z[events.dst], model=kind)
        )
    src, dst = dyad_indices(beta.shape[0])
    log_lam = log_intensity(beta[src], beta[dst], z[src], z[dst], model=kind)
    return event_term - t_max * jnp.sum(jnp.exp(log_lam))


def closed_form_constant_loglik(n_events: int, lam: float, t_max: float) -> float:
    """n_events log λ − T λ, the 2-node constant-intensity closed form."""
    return float(n_events) * float(jnp.log(lam)) - float(t_max) * float(lam)
