"""Classical row oracle for ``M_σ``.

``M_σ`` has a single 1 in each column, at row ``σ_j``. The row oracle, given
``ℓ``, returns the columns ``j`` with ``σ_j = ℓ``. Multiplicities are
``d_ℓ = |{j : σ_j = ℓ}|`` and ``d_τ = max d_ℓ``. Because
``M_σ M_σ^* = diag(d_ℓ)``, the spectral norm of ``M_σ`` is ``sqrt(d_τ)``.

Proposition 5.5 sets ``d_r = min{5, N}`` once ``N ||t - τ||_∞ < 1/(2π)``.
The quantum eight-candidate circuit that builds on that bound is not
implemented; this module is the classical lookup the bound is about.
"""

from __future__ import annotations

import math

from nuct._config import ensure_x64

ensure_x64()

import jax.numpy as jnp

from nuct.kernel import embedding_matrix

# Proposition 5.5. For N >= 5 this is 5; the min with N matters only for tiny N.
MULTIPLICITY_CAP = 5


def _as_ints(sigma) -> list[int]:
    return [int(s) for s in list(sigma)]


def row_support(sigma, ell: int) -> tuple[int, ...]:
    """Columns ``j`` with ``σ_j = ℓ``, in increasing ``j``."""
    ell = int(ell)
    return tuple(j for j, s in enumerate(_as_ints(sigma)) if s == ell)


def multiplicities(sigma, N: int) -> tuple[int, ...]:
    """``d_ℓ`` for ``ℓ = 0, ..., N-1``."""
    N = int(N)
    counts = [0] * N
    for s in _as_ints(sigma):
        if s < 0 or s >= N:
            raise ValueError(f"sigma entry {s} is outside 0..{N - 1}")
        counts[s] += 1
    return tuple(counts)


def max_multiplicity(sigma, N: int) -> int:
    """``d_τ = max_ℓ d_ℓ``. Zero when there are no columns."""
    counts = multiplicities(sigma, N)
    return max(counts) if counts else 0


def embedding_spectral_norm(sigma, N: int) -> float:
    """``sqrt(d_τ)``, the spectral norm of ``M_σ``."""
    return math.sqrt(max_multiplicity(sigma, N))


def numerical_embedding_spectral_norm(sigma, N: int) -> float:
    """Spectral norm of the explicit 0-1 matrix, for checking the formula."""
    M = embedding_matrix(sigma, N)
    return float(jnp.linalg.norm(M, ord=2))
