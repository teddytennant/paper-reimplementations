"""NISQ stand-ins for the hardware channel E in eq. 7.

A theoretical (unitary) DTQW never converges to the uniform prior; the paper
uses device noise as the mixing resource. We expose a unital depolarizing
channel (drives occupancy to uniform) and amplitude damping.
"""

from __future__ import annotations

import itertools

import numpy as np


def ket_to_rho(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1, 1)
    return psi @ psi.conj().T


def depolarize(rho: np.ndarray, p: float) -> np.ndarray:
    """E(ρ) = (1-p) ρ + p I/d, p in [0, 1]."""
    rho = np.asarray(rho, dtype=np.complex128)
    d = rho.shape[0]
    p = float(np.clip(p, 0.0, 1.0))
    return (1.0 - p) * rho + p * np.eye(d, dtype=np.complex128) / d


def _krons(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def amplitude_damping_kraus(gamma: float, n_qubits: int) -> list[np.ndarray]:
    """Product Kraus set for independent AD on each of n qubits."""
    g = float(np.clip(gamma, 0.0, 1.0))
    e0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - g)]], dtype=np.complex128)
    e1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=np.complex128)
    ops = []
    for bits in itertools.product((0, 1), repeat=n_qubits):
        ops.append(_krons([e0 if b == 0 else e1 for b in bits]))
    return ops


def apply_kraus(rho: np.ndarray, ops: list[np.ndarray]) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.complex128)
    out = np.zeros_like(rho)
    for e in ops:
        out = out + e @ rho @ e.conj().T
    return out


def amplitude_damping(rho: np.ndarray, gamma: float, n_qubits: int) -> np.ndarray:
    return apply_kraus(rho, amplitude_damping_kraus(gamma, n_qubits))
