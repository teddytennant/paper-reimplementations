"""Conditioning-free angle rounding, and the node-error operator bound.

Conditioning-free claim
-----------------------
The circuit feeds the exact stored offset ``z`` into ``arccos`` and rounds
only the output angle. If ``|φ̃ - arccos z| <= δ_φ``, then

    |T_ℓ(z) - cos(ℓ φ̃)| <= ℓ δ_φ

for every ``z ∈ [-1, 1]``, including points a hair inside the endpoints.
The bound does not contain ``κ(y) = 1/sqrt(1 - y^2)``. The analysis that
passes an input perturbation through ``arccos`` multiplies that perturbation
by ``κ``, which blows up as ``y → ±1``.

The rounding grid is not specified beyond the absolute tolerance. This
module rounds to the nearest multiple of ``2 δ_φ``, so the rounding error
is at most ``δ_φ`` (a tie lands on the boundary of the tolerance).

Node-error bound
----------------
The implementation brief calls this Eq. (28). In the extracted PDF the
operator-norm display is labeled Eq. (29), and the preceding entrywise
display is Eq. (28). The formula implemented here is the one the brief
states:

    ||F_t - F_τ||_2 <= (2π / sqrt(3)) N^{3/2} ||t - τ||_∞.
"""

from __future__ import annotations

import math

from nuct._config import ensure_x64

ensure_x64()

import jax.numpy as jnp

from nuct.chebyshev import nudft_matrix


def arccos_kappa(y) -> jnp.ndarray:
    """``κ(y) = 1/sqrt(1 - y^2) = |d arccos / dy|``. Undefined at ``±1``."""
    y = jnp.asarray(y, dtype=jnp.float64)
    return 1.0 / jnp.sqrt(1.0 - y * y)


def kappa_scaled_input_error(y, eps_in) -> jnp.ndarray:
    """First-order angle error of an input perturbation, ``κ(y) |ε_in|``.

    This is the factor the endpoint-unstable analysis carries. It is not a
    bound used by the NUCT circuit.
    """
    return arccos_kappa(y) * jnp.abs(jnp.asarray(eps_in, dtype=jnp.float64))


def round_arccos_output(z, delta_phi: float):
    """``arccos`` of the exact stored ``z``, then round only that angle.

    Returns ``(phi, phi_rounded)`` with ``|phi_rounded - phi| <= δ_φ``, up
    to a float64 ulp on the multiplication by the grid spacing.
    """
    delta_phi = float(delta_phi)
    if not math.isfinite(delta_phi) or delta_phi <= 0.0:
        raise ValueError(f"delta_phi must be positive and finite, got {delta_phi}")
    z = jnp.asarray(z, dtype=jnp.float64)
    phi = jnp.arccos(jnp.clip(z, -1.0, 1.0))
    quantum = 2.0 * delta_phi
    phi_rounded = jnp.round(phi / quantum) * quantum
    return phi, phi_rounded


def output_rounding_chebyshev_error(z, ell: int, delta_phi: float) -> dict:
    """``|T_ℓ(z) - cos(ℓ φ̃)|`` for angle-output rounding.

    ``bound`` is ``ℓ δ_φ``. It does not depend on ``z`` and does not contain
    ``κ``. ``tight_bound`` is ``ℓ`` times the realized angle error, which is
    at most ``δ_φ``.
    """
    ell = int(ell)
    if ell < 0:
        raise ValueError("ell must be nonnegative")
    phi, phi_rounded = round_arccos_output(z, delta_phi)
    exact = jnp.cos(ell * phi)
    approx = jnp.cos(ell * phi_rounded)
    angle_error = jnp.abs(phi_rounded - phi)
    return {
        "exact": exact,
        "approx": approx,
        "error": jnp.abs(approx - exact),
        "angle_error": angle_error,
        "bound": float(ell) * float(delta_phi),
        "tight_bound": float(ell) * angle_error,
    }


def worst_case_angle_chebyshev_error(z, ell: int, delta_phi: float) -> jnp.ndarray:
    """``|T_ℓ(z) - cos(ℓ (arccos z + δ_φ))|``, the Lipschitz endpoint of the bound.

    A perturbation of size exactly ``δ_φ`` is the worst output rounding the
    paper's hypothesis allows. The result is still at most ``ℓ δ_φ`` at
    ``z = 1 - 1e-12``, with no ``κ``.
    """
    ell = int(ell)
    delta_phi = float(delta_phi)
    z = jnp.asarray(z, dtype=jnp.float64)
    phi = jnp.arccos(jnp.clip(z, -1.0, 1.0))
    return jnp.abs(jnp.cos(ell * phi) - jnp.cos(ell * (phi + delta_phi)))


def realized_input_angle_error(y: float, eps_in: float) -> float:
    """``|arccos(y) - arccos(y_interior)|`` for an input step of size ``|eps_in|``.

    The step moves toward the interior, so a point near ``+1`` is decreased
    and a point near ``-1`` is increased. This is the angle error the
    input-perturbation analysis is about. Near an endpoint it tracks
    ``κ |ε|`` only while ``|ε|`` is small compared with the distance to
    ``±1``; a large step is even less like a ``κ``-free bound.
    """
    y = float(y)
    eps = abs(float(eps_in))
    if y >= 0.0:
        y2 = y - eps
    else:
        y2 = y + eps
    y2 = min(1.0, max(-1.0, y2))
    return abs(math.acos(min(1.0, max(-1.0, y))) - math.acos(y2))


def node_error_operator_bound(N: int, linf_node_error: float) -> float:
    """``(2π / sqrt(3)) N^{3/2} ||t - τ||_∞``.

    Spectral-norm bound on ``||F_t - F_τ||``. See the module docstring for
    the brief/PDF equation-number mismatch.
    """
    N = float(int(N))
    linf = float(linf_node_error)
    if N < 1 or linf < 0.0:
        raise ValueError("N must be positive and the node error nonnegative")
    return (2.0 * math.pi / math.sqrt(3.0)) * (N ** 1.5) * linf


def nudft_operator_distance(t, tau) -> float:
    """Spectral norm ``||F_t - F_τ||_2``."""
    diff = nudft_matrix(t) - nudft_matrix(tau)
    return float(jnp.linalg.norm(diff, ord=2))


def collision_hypothesis_holds(t, tau, N: int) -> bool:
    """Proposition 5.5 hypothesis: ``N ||t - τ||_∞ < 1/(2π)``."""
    t = jnp.asarray(t, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    margin = float(int(N) * jnp.max(jnp.abs(t - tau)))
    return margin < 1.0 / (2.0 * math.pi)


def minimum_node_bits(q: int) -> int:
    """Smallest ``m`` with ``N * 2^{-m} < 1/(2π)``.

    If ``|τ - t| <= 2^{-m}``, this ``m`` puts the quantization bin inside
    the Proposition 5.5 hypothesis. Realized error should still be checked:
    float64 rounding of ``arccos`` can add a fraction of an ulp.
    ``m = q + ceil(log2(2π))``.
    """
    q = int(q)
    if q < 1:
        raise ValueError("q must be at least 1")
    return q + math.ceil(math.log2(2.0 * math.pi))
