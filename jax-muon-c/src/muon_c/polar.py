"""Polar factors: exact thin SVD (paper Sec. 3) and complex Newton-Schulz (Appendix B)."""

from __future__ import annotations

import jax.numpy as jnp

NS_COEFFS = (3.4445, -4.775, 2.0315)
NS_STEPS = 5
EPS = 1e-8


def polar_svd(M: jnp.ndarray, *, tol: float = 1e-6) -> jnp.ndarray:
    """Canonical partial polar factor polar(M) = P Q^* from a compact SVD.

    For M = P Sigma Q^*, keep only singular values above a relative cutoff so
    that polar(0) = 0 and rank-deficient blocks omit null-space components
    (Appendix A). Batched over any leading axes. Last two axes are the matrix.
    """
    P, s, Qh = jnp.linalg.svd(M, full_matrices=False)
    smax = jnp.max(s, axis=-1, keepdims=True)
    mask = s > (tol * jnp.maximum(smax, 1e-12))
    return (P * mask[..., None, :]) @ Qh


def polar_ns(
    B: jnp.ndarray,
    *,
    steps: int = NS_STEPS,
    eps: float = EPS,
) -> jnp.ndarray:
    """Five-step Newton-Schulz polar from Appendix B.

    For each matrix block B, set Btilde = B^* if B has more rows than columns,
    else Btilde = B. Then

        X0 = Btilde / (||Btilde||_F + eps)
        A_j = X_j X_j^*
        X_{j+1} = a X_j + (b A_j + c A_j^2) X_j

    with (a, b, c) = (3.4445, -4.775, 2.0315), j = 0..4. Returns X5^* when the
    input was transposed, else X5. Computation is float32 or complex64.
    Leading axes are batch dimensions (frequency blocks, groups).
    """
    a, b, c = NS_COEFFS
    rows, cols = B.shape[-2], B.shape[-1]
    transposed = rows > cols
    Btilde = jnp.swapaxes(jnp.conj(B), -1, -2) if transposed else B
    if jnp.issubdtype(B.dtype, jnp.complexfloating):
        Btilde = Btilde.astype(jnp.complex64)
    else:
        Btilde = Btilde.astype(jnp.float32)
    fro = jnp.sqrt(jnp.sum(jnp.abs(Btilde) ** 2, axis=(-2, -1), keepdims=True))
    X = Btilde / (fro + eps)
    for _ in range(steps):
        A = X @ jnp.swapaxes(jnp.conj(X), -1, -2)
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = jnp.swapaxes(jnp.conj(X), -1, -2)
    return X
