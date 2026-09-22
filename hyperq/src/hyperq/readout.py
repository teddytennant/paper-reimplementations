"""Ring-plus-chord hypernetwork readout. JAX only."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class RingChord(NamedTuple):
    """Undirected ring-plus-chord graph.

    ``edge_list`` has shape ``(2n, 2)``: each undirected edge once.
    ``incident`` has shape ``(n, 4)``: edge index into ``edge_list`` for
    each qubit, in offset order ``+1, -1, +c, -c``.
    """

    edge_list: jax.Array
    incident: jax.Array


def edges(n: int, c: int) -> RingChord:
    """Undirected ring plus chord, degree 4, ``|E| = 2n``.

    Node ``i`` is linked to ``i ± 1`` and ``i ± c`` (mod ``n``).
    Requires ``gcd(c, n) == 1`` and four distinct offsets.
    Returns the edge list ``(2n, 2)`` and the per-qubit edge-index array
    ``(n, 4)`` so each qubit knows which edge couplings touch it.
    """
    n = int(n)
    c = int(c) % n
    if n < 5:
        raise ValueError("n must be >= 5 so ring and chord are 4-regular")
    if math.gcd(c, n) != 1:
        raise ValueError(f"gcd(c, n) must be 1, got gcd({c}, {n})")
    offs = (1, n - 1, c, (-c) % n)
    if len(set(offs)) != 4:
        raise ValueError("offsets ±1, ±c are not distinct; degree would not be 4")

    i = jnp.arange(n)
    ring = jnp.stack([i, (i + 1) % n], axis=1)
    chord = jnp.stack([i, (i + c) % n], axis=1)
    edge_list = jnp.concatenate([ring, chord], axis=0).astype(jnp.int32)

    # Same offset order as the neighbor list: +1, -1, +c, -c.
    plus1 = i
    minus1 = (i - 1) % n
    plus_c = n + i
    minus_c = n + ((i - c) % n)
    incident = jnp.stack([plus1, minus1, plus_c, minus_c], axis=1).astype(jnp.int32)
    return RingChord(edge_list, incident)


def init_params(key, n: int, d_h: int, d_a: int, d_b: int | None = None):
    """Parameters consumed by ``emit``.

    ``b = W_A @ h`` has width ``d_a`` (``W_A`` is ``(d_a, d_h)``).
    ``W_alpha`` is ``(n, d_a)``, ``W_omega`` is ``(|E|, d_a)`` with
    ``|E| = 2n``, and ``W_psi`` is ``(2n, d_a)`` so ``W_psi @ b`` has
    shape ``(2n,)`` (yz coordinates). There is no free bias ``b`` and no
    free ``yz``. ``d_b`` is accepted only so older call sites still run;
    it is not a parameter dimension.
    """
    del d_b
    n = int(n)
    n_edges = 2 * n
    k_a, k_al, k_w, k_psi = jax.random.split(key, 4)
    scale = 0.02
    return {
        "W_A": scale * jax.random.normal(k_a, (d_a, d_h)),
        "W_alpha": scale * jax.random.normal(k_al, (n, d_a)),
        "W_omega": scale * jax.random.normal(k_w, (n_edges, d_a)),
        "W_psi": scale * jax.random.normal(k_psi, (2 * n, d_a)),
    }


def _unit_psi(yz):
    y = yz[:, 0]
    z = yz[:, 1]
    nrm = jnp.sqrt(y * y + z * z)
    zero = nrm == 0
    nrm = jnp.where(zero, jnp.ones_like(nrm), nrm)
    psi_y = jnp.where(zero, jnp.zeros_like(y), y / nrm)
    psi_z = jnp.where(zero, jnp.ones_like(z), z / nrm)
    return psi_y, psi_z


def emit(h, params):
    """Emit angles and unit psi, all depending on ``h``.

    ``b = W_A @ h``
    ``alpha = pi * tanh(W_alpha @ b)`` with shape ``(n,)``
    ``omega = 1.2 * tanh(W_omega @ b)`` with shape ``(|E|,)``
    ``psi`` is the per-qubit unit vector of ``W_psi @ b`` (shape ``(2n,)``).
    """
    b = params["W_A"] @ h
    alpha = jnp.pi * jnp.tanh(params["W_alpha"] @ b)
    omega = 1.2 * jnp.tanh(params["W_omega"] @ b)
    n = alpha.shape[0]
    raw = (params["W_psi"] @ b).reshape((n, 2))
    return alpha, omega, _unit_psi(raw)


def _incident_index(edges) -> jax.Array:
    """Neighbor-edge index array, shape ``(n, 4)``."""
    if isinstance(edges, RingChord):
        return edges.incident
    return edges


def _cos(x):
    """Cosine of an edge coupling. Overridable in tests that need exact 0."""
    return jnp.cos(x)


def readout(alpha, omega, psi, edges):
    """``z_q = (psi_z cos α - psi_y sin α) * ∏_{e ∋ q} cos(ω_e)``.

    The four factors are the cosines of the edge couplings incident to
    ``q``, looked up by edge index. Four explicit multiplies, not
    ``jnp.prod``, so the Jacobian stays finite when one cosine is exactly
    zero. Identity ``α = 0, ω = 0`` gives ``z = psi_z``.
    ``psi`` is ``(psi_y, psi_z)``. ``edges`` is the ``(n, 4)`` edge-index
    array (or a ``RingChord``).
    """
    psi_y, psi_z = psi
    amp = psi_z * jnp.cos(alpha) - psi_y * jnp.sin(alpha)
    incident = _incident_index(edges)
    # Index omega by edge id, never by neighbor node id.
    coupled = omega[incident]
    c0 = _cos(coupled[:, 0])
    c1 = _cos(coupled[:, 1])
    c2 = _cos(coupled[:, 2])
    c3 = _cos(coupled[:, 3])
    return amp * c0 * c1 * c2 * c3


def residual(z, W_B, s, r):
    """``(s / r) * W_B @ z``."""
    return (s / r) * (W_B @ z)
