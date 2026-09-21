"""Prisoner's Dilemma mini-experiment 1 (paper Section VII, eqs 34-36)."""

from __future__ import annotations

import jax.numpy as jnp

from mefpia.payoff import as_tensor

Array = jnp.ndarray

# Classical PD outcome rewards (eq 34), ordered (CC, CD, DC, DD):
# r1 = (3, 0, 5, 1), r2 = (3, 5, 0, 1).
_R1_CLASSICAL = (3.0, 0.0, 5.0, 1.0)
_R2_CLASSICAL = (3.0, 5.0, 0.0, 1.0)

# Mutual defection is the unique Nash of the classical game, payoff 1 each.
MUTUAL_DEFECTION_PAYOFF = 1.0


def _computational_projectors(dtype) -> list[Array]:
    """Separable projective POVM on C^2 ⊗ C^2.

    Cooperate = |0><0|, defect = |1><1| for each player. Joint outcomes
    follow eqs 35-36:
        P_omega1 = |00><00| (CC), P_omega2 = |01><01| (CD),
        P_omega3 = |10><10| (DC), P_omega4 = |11><11| (DD).
    """
    projectors = []
    for a in range(2):
        for b in range(2):
            ket = jnp.zeros((4,), dtype=dtype)
            ket = ket.at[2 * a + b].set(1)
            projectors.append(jnp.outer(ket, jnp.conjugate(ket)))
    return projectors


def prisoners_dilemma(dtype=jnp.complex128) -> dict[str, Array | tuple[int, int]]:
    """Build EGW reward tensors for the PD mapping of mini-experiment 1.

    Reward operators are R_i = sum_omega r_i(omega) P_omega (eq 16) with the
    projective POVM above. Returned tensors have shape (2, 2, 2, 2) so the
    hot path never materializes a joint density matrix.
    """
    povm = _computational_projectors(dtype)
    r1 = jnp.zeros((4, 4), dtype=dtype)
    r2 = jnp.zeros((4, 4), dtype=dtype)
    for p, v1, v2 in zip(povm, _R1_CLASSICAL, _R2_CLASSICAL):
        r1 = r1 + v1 * p
        r2 = r2 + v2 * p
    dims = (2, 2)
    return {
        "dims": dims,
        "rewards": (as_tensor(r1, dims), as_tensor(r2, dims)),
        "r1_matrix": r1,
        "r2_matrix": r2,
        "r_star": jnp.asarray(MUTUAL_DEFECTION_PAYOFF),
    }


def relative_payoff_error(r: Array, r_star: Array) -> Array:
    """epsilon_rel = |r - r_star| / |r_star| (eq 33)."""
    return jnp.abs(r - r_star) / jnp.abs(r_star)
