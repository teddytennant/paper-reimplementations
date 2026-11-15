"""Optimizer-state scaling identity from Section 2.

Full Adam scales as Psi * K_opt. RoSA scales as (l/L) * Psi * rho * K_opt,
where l is the number of active layers (|S_e|), L is depth, rho is the
sparsity density, Psi is the parameter count, and K_opt is bytes per
parameter for optimizer states. This is a scaling identity, not a measured
VRAM number.
"""

from __future__ import annotations


def adam_optimizer_bytes(psi, k_opt):
    """Full Adam optimizer-state scale: Psi * K_opt."""

    return psi * k_opt


def rosa_optimizer_bytes(psi, k_opt, active_layers, depth, rho):
    """RoSA optimizer-state scale: (l/L) * Psi * rho * K_opt."""

    if depth == 0:
        raise ValueError("depth L must be positive")
    return (active_layers / depth) * psi * rho * k_opt
