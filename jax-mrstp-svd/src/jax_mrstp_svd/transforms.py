"""Invertible linear transforms L for the generalized t-product (paper eq. (1)).

L(A) = A ×_3 L.  FFT (unnormalized DFT along tubes) is the default, matching
Section 6. Identity and an explicit DFT matrix are provided for tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


def _mode3(A: Array, Lmat: Array) -> Array:
    """Mode-3 product: each tube t is replaced by Lmat @ t."""
    return jnp.einsum("kl,ijl->ijk", Lmat, A)


def dft_matrix(n: int, dtype=jnp.complex128) -> Array:
    """Unnormalized DFT matrix F_{kj} = exp(-2πi kj / n).  F^H F = n I (ρ = n)."""
    k = jnp.arange(n)
    F = jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / n)
    return F.astype(dtype)


@dataclass(frozen=True)
class Transform:
    """Named invertible transform along tubes (axis=2)."""

    name: str

    def forward(self, A: Array) -> Array:
        raise NotImplementedError

    def inverse(self, A_bar: Array) -> Array:
        raise NotImplementedError

    def rho(self, n3: int) -> float:
        """Constant ρ in L^H L = ρ I (Theorem 4.3 / 5.1)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Identity(Transform):
    """L = I.  Real arithmetic; ρ = 1."""

    name: str = "identity"

    def forward(self, A: Array) -> Array:
        return A

    def inverse(self, A_bar: Array) -> Array:
        return A_bar

    def rho(self, n3: int) -> float:
        return 1.0


@dataclass(frozen=True)
class FFT(Transform):
    """Unnormalized FFT along tubes.  Equivalent to DFT; ρ = n3."""

    name: str = "fft"

    def forward(self, A: Array) -> Array:
        return jnp.fft.fft(A, axis=2)

    def inverse(self, A_bar: Array) -> Array:
        return jnp.fft.ifft(A_bar, axis=2)

    def rho(self, n3: int) -> float:
        return float(n3)


@dataclass(frozen=True)
class DFT(Transform):
    """Explicit DFT matrix applied as a mode-3 product.  ρ = n3."""

    name: str = "dft"

    def forward(self, A: Array) -> Array:
        n3 = A.shape[2]
        return _mode3(A, dft_matrix(n3, dtype=jnp.complex128))

    def inverse(self, A_bar: Array) -> Array:
        n3 = A_bar.shape[2]
        F = dft_matrix(n3, dtype=jnp.complex128)
        return _mode3(A_bar, jnp.conj(F).T / n3)

    def rho(self, n3: int) -> float:
        return float(n3)


_REGISTRY = {
    "identity": Identity(),
    "fft": FFT(),
    "dft": DFT(),
}


def get_transform(spec: str | Transform) -> Transform:
    if isinstance(spec, Transform):
        return spec
    key = spec.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown transform {spec!r}. Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[key]
