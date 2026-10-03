"""Integer split of a fixed-point node (Eq. (6), Lemma 4.2).

``τ = a / 2^m`` with ``a`` an integer. ``N = 2^q``. The split is exact bit
arithmetic: ``round(N τ)`` is decided from the integer bits of ``a``, never
by rounding a float of ``τ``.

Tie convention
--------------
Lemma 4.2 says ``round`` uses a fixed tie convention and does not name it.
This module uses **round half away from zero**. For ``τ >= 0`` that is
round-half-up: if ``N τ`` is a half-integer, ``sunw = floor(N τ) + 1`` and
``z = -1``. Round-half-to-even also keeps ``|z| <= 1``; it is not used, and
the formula is not switched in order to satisfy the multiplicity claim.

``sunw`` lies in ``{0, ..., N}``. When ``τ`` is within ``1/(2N)`` of 1,
``sunw = N`` and ``σ = sunw mod N = 0``. The boundary ``τ = 1 - 1/(2N)`` is
a tie and falls under the convention above, which also yields ``sunw = N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from nuct._config import ensure_x64

ensure_x64()

import jax.numpy as jnp

from nuct.chebyshev import equispaced_nodes, nuct_order, nuct_times

# Pinned so tests and callers can name the convention without reading the proof.
TIE_CONVENTION = "half_away_from_zero"

# Float64 can represent 2^m exactly only for m <= 53. Quantizing an irrational
# time, and the float view of τ, are limited to that range. The integer split
# itself is not.
_MAX_FLOAT_BITS = 53


@dataclass(frozen=True)
class NodeSplit:
    """One node after the Eq. (6) split.

    ``z = z_numerator / z_denominator`` is a dyadic in ``[-1, 1]``. When
    ``m >= q + 1`` the denominator is ``2^{m-q-1}``, the bit width in the
    paper. It is not reduced, so that width stays visible. When ``m <= q``,
    ``N τ`` is an integer, ``z = 0`` and the denominator is 1.
    """

    a: int
    m: int
    q: int
    sunw: int
    sigma: int
    z_numerator: int
    z_denominator: int

    @property
    def N(self) -> int:
        return 1 << self.q

    @property
    def z(self) -> float:
        """Float64 view of ``z``. Exact for ``m <= 53``."""
        return self.z_numerator / self.z_denominator

    def z_fraction(self) -> Fraction:
        return Fraction(self.z_numerator, self.z_denominator)

    def tau_fraction(self) -> Fraction:
        return Fraction(self.a, 1 << self.m)

    def reconstructed_N_tau(self) -> Fraction:
        """``sunw + z/2``, which equals ``N τ`` exactly."""
        return self.sunw + self.z_fraction() / 2


def split_fixed_point(a: int, m: int, q: int) -> NodeSplit:
    """Split ``τ = a / 2^m`` against ``N = 2^q``.

    ``a`` must be an integer in ``{0, ..., 2^m}`` so ``τ ∈ [0, 1]``. The
    paper's open interval is ``[0, 1)``; ``τ = 1`` is accepted as a boundary
    and maps to ``sunw = N``, ``σ = 0``, ``z = 0``.

    Returns ``sunw = round(N τ)``, ``σ = sunw mod N``, and ``z`` with
    ``N τ = sunw + z/2`` and ``|z| <= 1``.
    """
    a = int(a)
    m = int(m)
    q = int(q)
    if m < 0 or q < 0:
        raise ValueError(f"m and q must be nonnegative, got m={m}, q={q}")
    if a < 0 or a > (1 << m):
        raise ValueError(f"a must lie in {{0, ..., 2^m}}, got a={a}, m={m}")

    N = 1 << q
    if m > q:
        p = m - q
        modulus = 1 << p
        half = 1 << (p - 1)
        quotient = a >> p
        remainder = a & (modulus - 1)
        if remainder < half:
            sunw = quotient
        elif remainder > half:
            sunw = quotient + 1
        else:
            # Tie. Half away from zero, and τ >= 0, so half up.
            sunw = quotient + 1
        z_numerator = a - sunw * modulus
        z_denominator = half
    elif m == q:
        sunw = a
        z_numerator = 0
        z_denominator = 1
    else:
        sunw = a << (q - m)
        z_numerator = 0
        z_denominator = 1

    if not 0 <= sunw <= N:
        raise RuntimeError(f"split invariant failed: sunw={sunw} not in 0..{N}")
    if z_denominator <= 0 or abs(z_numerator) > z_denominator:
        raise RuntimeError(
            f"split invariant failed: z={z_numerator}/{z_denominator} not in [-1, 1]"
        )
    return NodeSplit(
        a=a,
        m=m,
        q=q,
        sunw=sunw,
        sigma=sunw % N,
        z_numerator=z_numerator,
        z_denominator=z_denominator,
    )


def split_phase_sides(k: int, split: NodeSplit):
    """Both sides of the Eq. (6) phase identity.

    ``exp(-2π i k τ) = exp(-2π i k σ / N) exp(-i π (k/N) z)``.

    The float view of ``τ`` is exact only for ``m <= 53``. The algebraic
    identity ``N τ = sunw + z/2`` does not use this function; compare
    :meth:`NodeSplit.reconstructed_N_tau` for that.
    """
    if split.m > _MAX_FLOAT_BITS:
        raise ValueError(f"float64 phase check needs m <= {_MAX_FLOAT_BITS}")
    N = split.N
    tau = split.a / float(1 << split.m)
    z = split.z
    left = jnp.exp(-2j * jnp.pi * k * tau)
    right = jnp.exp(-2j * jnp.pi * k * split.sigma / N) * jnp.exp(
        -1j * jnp.pi * (k / N) * z
    )
    return left, right


def quantize_unit_interval(t: float, m: int) -> int:
    """Round ``t ∈ [0, 1]`` to an ``m``-bit fixed point, half away from zero.

    This is the classical stand-in for the node oracle, not the split.
    ``2^m`` must be a float64 integer, so ``0 <= m <= 53``. ``t = 1`` maps
    to ``a = 2^m``. Half-integers of ``t * 2^m`` round up.

    A float64 value of an irrational ``t`` can sit a fraction of an ulp off
    the true ``t * 2^m``. Callers that need ``|τ - t| <= 2^{-m}`` should
    check the realized error. For the NUCT nodes at the sizes tested here,
    that check passes.
    """
    m = int(m)
    if m < 0 or m > _MAX_FLOAT_BITS:
        raise ValueError(f"float64 quantization needs 0 <= m <= {_MAX_FLOAT_BITS}")
    t = float(t)
    if t < 0.0 or t > 1.0:
        raise ValueError(f"t must lie in [0, 1], got {t}")
    if t == 1.0:
        return 1 << m
    if t == 0.0:
        return 0
    # floor(scaled + 1/2) is half-up for a nonnegative scaled value.
    # 2^m is exact in float64 for m <= 53.
    scaled = t * float(1 << m)
    a = int(jnp.floor(jnp.asarray(scaled, dtype=jnp.float64) + 0.5))
    if a < 0:
        return 0
    if a > (1 << m):
        return 1 << m
    return a


def store_nuct_nodes(N: int, m: int) -> dict:
    """``m``-bit stored nodes for the equispaced-in-``x`` times of Definition 1.1.

    ``k = 0`` is stored as the exact dyadic ``1/2`` (Proposition 5.2). Every
    other time is quantized with :func:`quantize_unit_interval` and clipped
    to ``[0, 1/2]``, then split with :func:`split_fixed_point`. Requires
    ``q + 1 <= m <= 53``, the range in which the paper's node oracle has a
    fractional bit and float64 can hold ``2^m``.
    """
    q = nuct_order(N)
    m = int(m)
    if m < q + 1 or m > _MAX_FLOAT_BITS:
        raise ValueError(f"stored NUCT nodes need q+1 <= m <= 53, got q={q}, m={m}")
    x = equispaced_nodes(N)
    t = nuct_times(x).at[0].set(0.5)
    half = 1 << (m - 1)
    a_values: list[int] = []
    splits: list[NodeSplit] = []
    for k in range(N):
        if k == 0:
            a_k = half
        else:
            a_k = min(quantize_unit_interval(float(t[k]), m), half)
        a_values.append(a_k)
        splits.append(split_fixed_point(a_k, m, q))
    tau = jnp.asarray([s.a / float(1 << m) for s in splits], dtype=jnp.float64)
    return {
        "N": N,
        "q": q,
        "m": m,
        "x": x,
        "t": t,
        "a": a_values,
        "splits": splits,
        "sunw": jnp.asarray([s.sunw for s in splits], dtype=jnp.int32),
        "sigma": jnp.asarray([s.sigma for s in splits], dtype=jnp.int32),
        "z": jnp.asarray([s.z for s in splits], dtype=jnp.float64),
        "tau": tau,
        "linf_error": float(jnp.max(jnp.abs(t - tau))),
    }
