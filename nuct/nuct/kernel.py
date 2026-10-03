"""Bessel-Chebyshev factorization of the interaction kernel, and Eq. (31).

The kernel is ``H(x, z) = exp(-i π x z / 2)`` on ``[-1, 1]^2``. Its
Chebyshev coefficients are

    α_{ℓ r} = 4 i^r J_{(ℓ+r)/2}(-π/4) J_{(r-ℓ)/2}(-π/4)

when ``ℓ ≡ r (mod 2)``, and 0 otherwise, then ``α'_{ℓ r} = η_ℓ η_r α_{ℓ r}``
with ``η_0 = 1/2`` and ``η_n = 1`` for ``n >= 1``. The rank-``K`` truncation
is the partial sum of ``α'_{ℓ r} T_ℓ(z) T_r(x)``.

``K`` is an argument. The paper's Lambert-W schedule
``K = O(log(1/ε) / log log(1/ε))`` is not hard-coded: a wrong closed form
would be worse than passing ``K`` in.

Negative Bessel orders use ``J_{-n}(z) = (-1)^n J_n(z)``.
"""

from __future__ import annotations

from nuct._config import ensure_x64

ensure_x64()

import jax.numpy as jnp
from jax.scipy.special import bessel_jn

from nuct.chebyshev import chebyshev_T, unitary_dft


def interaction_kernel(x, z) -> jnp.ndarray:
    """``H(x, z) = exp(-i π x z / 2)``. ``x`` and ``z`` broadcast."""
    x = jnp.asarray(x, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    return jnp.exp(-1j * jnp.pi * x * z / 2.0)


def bessel_coeff_table(K: int) -> jnp.ndarray:
    """``α'_{ℓ r}`` for ``0 <= ℓ, r < K``, shape ``(K, K)``, complex128.

    Axis 0 is ``ℓ`` (the ``z`` degree). Axis 1 is ``r`` (the ``x`` degree).
    """
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")
    J = bessel_jn(-jnp.pi / 4.0, v=K - 1)
    ell = jnp.arange(K)[:, None]
    r = jnp.arange(K)[None, :]
    same_parity = ((ell + r) % 2) == 0
    order_plus = (ell + r) // 2
    order_minus = (r - ell) // 2
    J_plus = J[order_plus]
    order_abs = jnp.abs(order_minus)
    negative_odd = (order_minus < 0) & ((order_abs % 2) == 1)
    J_minus = jnp.where(negative_odd, -J[order_abs], J[order_abs])
    alpha = 4.0 * ((1j) ** r) * J_plus * J_minus
    eta_ell = jnp.where(ell == 0, 0.5, 1.0)
    eta_r = jnp.where(r == 0, 0.5, 1.0)
    table = jnp.where(same_parity, eta_ell * eta_r * alpha, 0j)
    return jnp.asarray(table, dtype=jnp.complex128)


def truncated_kernel(x, z, coeffs) -> jnp.ndarray:
    """``H_K(x, z) = sum_{ℓ,r} α'_{ℓ r} T_ℓ(z) T_r(x)``.

    ``x`` and ``z`` are 1-d. The result has shape ``(len(z), len(x))``,
    so entry ``[j, i]`` is ``H_K(x_i, z_j)``.
    """
    x = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
    z = jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64))
    coeffs = jnp.asarray(coeffs)
    K = int(coeffs.shape[0])
    degree = jnp.arange(K)
    T_x = chebyshev_T(degree[:, None], x[None, :])
    T_z = chebyshev_T(degree[:, None], z[None, :])
    return jnp.einsum("er,ej,ri->ji", coeffs, T_z, T_x)


def frequency_coordinates(N: int) -> jnp.ndarray:
    """``w_k = 2k/N - 1 ∈ [-1, 1]``, the rescaled frequency of Eq. (14)."""
    if int(N) < 1:
        raise ValueError("N must be positive")
    k = jnp.arange(N, dtype=jnp.float64)
    return 2.0 * k / jnp.asarray(N, dtype=jnp.float64) - 1.0


def frequency_phase_rescaling(k: int, z, N: int):
    """Both sides of Eq. (14).

    ``exp(-i π (k/N) z) = exp(-i π z / 2) H(w_k, z)``,
    with ``w_k = 2k/N - 1``.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    w = 2.0 * int(k) / int(N) - 1.0
    left = jnp.exp(-1j * jnp.pi * (int(k) / int(N)) * z)
    right = jnp.exp(-1j * jnp.pi * z / 2.0) * interaction_kernel(w, z)
    return left, right


def frequency_weights(N: int, K: int) -> jnp.ndarray:
    """``v_r(k) = T_r(w_k)``, shape ``(K, N)``."""
    w = frequency_coordinates(N)
    degree = jnp.arange(int(K))[:, None]
    return chebyshev_T(degree, w[None, :])


def node_amplitudes(z, coeffs) -> jnp.ndarray:
    """``u_r(j) = exp(-i π z_j / 2) sum_ℓ α'_{ℓ r} T_ℓ(z_j)``, shape ``(K, n)``.

    ``z_j`` is the exact stored offset. Nothing here differentiates
    ``arccos`` or multiplies by ``κ``.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    coeffs = jnp.asarray(coeffs)
    K = int(coeffs.shape[0])
    degree = jnp.arange(K)[:, None]
    T_z = chebyshev_T(degree, z[None, :])
    core = jnp.einsum("er,ej->rj", coeffs, T_z)
    phase = jnp.exp(-1j * jnp.pi * z / 2.0)
    return core * phase[None, :]


def embedding_matrix(sigma, N: int) -> jnp.ndarray:
    """``M_σ`` with a single 1 at row ``σ_j``, column ``j``. Shape ``(N, n)``."""
    sigma_list = [int(s) for s in list(sigma)]
    n = len(sigma_list)
    if any(s < 0 or s >= int(N) for s in sigma_list):
        raise ValueError(f"sigma entries must lie in 0..{int(N) - 1}")
    M = jnp.zeros((int(N), n), dtype=jnp.float64)
    if n == 0:
        return M
    rows = jnp.asarray(sigma_list, dtype=jnp.int32)
    cols = jnp.arange(n, dtype=jnp.int32)
    return M.at[rows, cols].set(1.0)


def factorized_nudft(z, sigma, K: int, coeffs=None) -> jnp.ndarray:
    """``F_τ^{(K)} = sum_{r < K} diag(v_r) F M_σ diag(u_r)`` (Eq. (31)).

    Square case: ``len(z) = len(sigma) = N``, and ``F`` is the ``N``-point
    unitary DFT. ``K`` may exceed ``N``; the rank is not the DFT length.
    The matrix this approaches, as ``K`` grows, is the NUDFT of the **stored**
    nodes ``τ``, not the NUDFT of the exact angles ``t``.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    sigma_list = [int(s) for s in list(sigma)]
    N = int(z.shape[0])
    if len(sigma_list) != N:
        raise ValueError("z and sigma must have the same length")
    K = int(K)
    if K < 1:
        raise ValueError("K must be at least 1")
    if coeffs is None:
        coeffs = bessel_coeff_table(K)
    else:
        coeffs = jnp.asarray(coeffs)
        if int(coeffs.shape[0]) < K or int(coeffs.shape[1]) < K:
            raise ValueError("coeffs must be at least K x K")
        coeffs = coeffs[:K, :K]
    F = unitary_dft(N)
    M = embedding_matrix(sigma_list, N)
    v = frequency_weights(N, K)
    u = node_amplitudes(z, coeffs)
    acc = jnp.zeros((N, N), dtype=jnp.complex128)
    for r in range(K):
        acc = acc + jnp.diag(v[r]) @ F @ M @ jnp.diag(u[r])
    return acc
