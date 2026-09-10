"""Muon-C direction, unfolded Muon, and one Algorithm 1 kernel step."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from muon_c.geometry import critical_fft, critical_ifft, rms, unfold
from muon_c.polar import EPS, polar_ns, polar_svd

BETA = 0.95
TARGET_RMS = 0.2


def _polar_fn(kind: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if kind in ("exact", "svd"):
        return polar_svd
    if kind in ("ns", "newton_schulz"):
        return polar_ns
    raise ValueError(f"unknown polar kind: {kind}")


def _grouped(kernel: jnp.ndarray, groups: int) -> jnp.ndarray:
    cout, cin_g, kh, kw = kernel.shape
    if cout % groups != 0:
        raise ValueError("Cout must be divisible by groups")
    return kernel.reshape(groups, cout // groups, cin_g, kh, kw)


def muon_c_direction(
    M: jnp.ndarray,
    *,
    groups: int = 1,
    polar: str = "exact",
) -> jnp.ndarray:
    """Unscaled Muon-C direction O on a Conv2d kernel (Section 5.2).

    Layout is (Cout, Cin, kh, kw), or grouped (Cout, Cin/g, kh, kw). Polarizes
    each frequency block independently, then inverse orthonormal DFT. For real
    momentum the result is real (imaginary part dropped after ifft).
    """
    polarize = _polar_fn(polar)
    G = _grouped(M, groups)
    # (..., Cout/g, Cin/g, kh, kw) -> (..., kh, kw, Cout/g, Cin/g) so the last
    # two axes are the channel-transfer matrix at each frequency.
    Mhat = critical_fft(G)
    blocks = jnp.moveaxis(Mhat, (-4, -3), (-2, -1))
    Ohat_blocks = polarize(blocks)
    Ohat = jnp.moveaxis(Ohat_blocks, (-2, -1), (-4, -3))
    O = critical_ifft(Ohat)
    O = O.reshape(M.shape)
    if not jnp.issubdtype(M.dtype, jnp.complexfloating):
        O = jnp.real(O)
    return O.astype(M.dtype)


def muon_c_scale(O: jnp.ndarray, *, eps: float = EPS) -> jnp.ndarray:
    """s_f = 0.2 / (RMS(O) + eps) from Algorithm 1 / Section 5.4."""
    return TARGET_RMS / (rms(O) + eps)


def unfolded_muon_direction(M: jnp.ndarray, *, polar: str = "exact") -> jnp.ndarray:
    """Global polar of the Cout x (Cin kh kw) unfolding (unfolded Muon)."""
    polarize = _polar_fn(polar)
    cout, cin, kh, kw = M.shape
    B = unfold(M)
    P = polarize(B)
    return P.reshape(cout, cin, kh, kw).astype(M.dtype)


def unfolded_scale(kernel: jnp.ndarray) -> jnp.ndarray:
    """s_g = 0.2 * sqrt(max{Cout, n Cin}) from Section 5.4."""
    cout, cin, kh, kw = kernel.shape
    n = kh * kw
    return TARGET_RMS * jnp.sqrt(jnp.maximum(float(cout), float(n * cin)))


def nesterov_lookahead(
    momentum: jnp.ndarray,
    grad: jnp.ndarray,
    *,
    beta: float = BETA,
    nesterov: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """EMA momentum and optional Nesterov look-ahead (Algorithm 1, lines 1-2)."""
    M = beta * momentum + (1.0 - beta) * grad
    shaped = beta * M + (1.0 - beta) * grad if nesterov else M
    return M, shaped


def muon_c_step(
    weight: jnp.ndarray,
    grad: jnp.ndarray,
    momentum: jnp.ndarray,
    *,
    eta: float,
    weight_decay: float = 0.0,
    beta: float = BETA,
    nesterov: bool = True,
    polar: str = "ns",
    groups: int = 1,
    eps: float = EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One operator-aligned Muon-C update on a finite convolutional kernel.

    Algorithm 1: momentum, critical-grid FFT, blockwise polar, inverse FFT,
    RMS scale, decoupled weight decay. Intended for Conv2d kernels with
    kh * kw > 1. Returns (W^{t+1}, M^t).
    """
    M, shaped = nesterov_lookahead(momentum, grad, beta=beta, nesterov=nesterov)
    O = muon_c_direction(shaped, groups=groups, polar=polar)
    s_f = muon_c_scale(O, eps=eps)
    new_weight = (1.0 - eta * weight_decay) * weight - eta * s_f * O
    return new_weight, M


def uses_muon_c(kernel: jnp.ndarray) -> bool:
    """Routing: spatial Conv2d kernels with kh * kw > 1 use Muon-C."""
    return kernel.ndim == 4 and (kernel.shape[-2] * kernel.shape[-1] > 1)
