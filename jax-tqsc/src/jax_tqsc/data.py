"""Synthetic 13-channel HVDR+aux records (stand-in for the paper's coincidence vector).

This is not a recreation of the SPDC/MMF experiment. Channels 0–3 are noisy
coincidence-like projections onto H, V, D, R; 4–7 Alice singles; 8–11 Bob
singles; 12 a dummy coincidence-time. Fixed crosstalk plus Poisson/Gaussian
noise make a linear-inversion baseline imperfect.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# |H>=|0>, |V>=|1>, |D>=(|0>+|1>)/sqrt(2), |R>=(|0>+i|1>)/sqrt(2)
_H = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
_V = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex64)
_D = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex64)
_R = 0.5 * np.array([[1.0, -1j], [1j, 1.0]], dtype=np.complex64)
PROJECTORS = np.stack([_H, _V, _D, _R], axis=0)

# Mild, fixed readout crosstalk among the four coincidence channels.
_CROSSTALK = np.array(
    [
        [0.92, 0.04, 0.03, 0.01],
        [0.03, 0.91, 0.04, 0.02],
        [0.02, 0.03, 0.90, 0.05],
        [0.03, 0.02, 0.04, 0.91],
    ],
    dtype=np.float64,
)


def _haar_pure(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    psi = z / np.linalg.norm(z, axis=-1, keepdims=True)
    return (psi[:, :, None] * np.conjugate(psi[:, None, :])).astype(np.complex64)


def _born(rho: np.ndarray) -> np.ndarray:
    # Tr(rho P_k) for k in {H,V,D,R}; rho (n,2,2), PROJECTORS (4,2,2)
    return np.real(np.einsum("nij,kji->nk", rho, PROJECTORS)).astype(np.float64)


def make_synthetic_dataset(
    n: int = 256,
    seed: int = 0,
    n_coin: float = 200.0,
    n_alice: float = 800.0,
    n_bob: float = 600.0,
    extra_gauss: float = 0.04,
    p_mix_max: float = 0.7,
) -> dict[str, Any]:
    """Random Bloch-ball states and noisy 13-vectors.

    rho = (1-p)|ψ⟩⟨ψ| + p I/2 with Haar |ψ⟩ and p=0 on half the samples.
    """
    rng = np.random.default_rng(seed)
    rho_pure = _haar_pure(rng, n)
    p = rng.uniform(0.0, p_mix_max, size=(n,))
    p[: n // 2] = 0.0
    eye = np.eye(2, dtype=np.complex64) * 0.5
    rho = ((1.0 - p)[:, None, None] * rho_pure + p[:, None, None] * eye).astype(np.complex64)

    probs = np.clip(_born(rho), 1e-6, 1.0)
    lam_c = n_coin * (probs @ _CROSSTALK.T)
    lam_a = n_alice * probs
    lam_b = n_bob * (0.85 * probs + 0.15 * 0.5)

    coinc = rng.poisson(lam_c).astype(np.float64)
    alice = rng.poisson(lam_a).astype(np.float64)
    bob = rng.poisson(lam_b).astype(np.float64)
    coinc = coinc + rng.normal(0.0, extra_gauss * n_coin, size=coinc.shape)
    alice = alice + rng.normal(0.0, extra_gauss * n_alice, size=alice.shape)
    bob = bob + rng.normal(0.0, extra_gauss * n_bob, size=bob.shape)
    coinc = np.clip(coinc, 0.0, None)
    alice = np.clip(alice, 0.0, None)
    bob = np.clip(bob, 0.0, None)

    total_c = coinc.sum(axis=1, keepdims=True)
    tau = 1.0 + 0.15 * (total_c / (4.0 * n_coin) - 1.0) + rng.normal(0.0, 0.05, size=(n, 1))

    m = np.concatenate(
        [coinc / n_coin, alice / n_alice, bob / n_bob, tau],
        axis=1,
    ).astype(np.float32)
    return {"M": m, "rho": rho, "p": p.astype(np.float32), "psi_rho": rho_pure}
