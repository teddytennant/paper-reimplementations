"""Stiefel geometry used by Stiefel-AdamW (arXiv:2609.21039, Appendix A).

Column-orthonormal Stiefel: St(n, r) = {X ∈ R^{n×r} : X^T X = I_r}.
The paper's factor A ∈ R^{r×n} is row-orthonormal (A A^T = I_r); we work
with X := A^T ∈ St(n, r) and recover A = X^T (Appendix A.1.2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def skew(m: Array) -> Array:
    """Skew-symmetric part: (M - M^T) / 2."""
    return 0.5 * (m - m.T)


def sym(m: Array) -> Array:
    """Symmetric part: (M + M^T) / 2."""
    return 0.5 * (m + m.T)


def project_a1(x: Array, z: Array) -> Array:
    """Orthogonal projection onto T_X St(n, r), formula (A.1).

    P_X(Z) = X skew(X^T Z) + (I - X X^T) Z.
    """
    xtz = x.T @ z
    return x @ skew(xtz) + z - x @ (x.T @ z)


def project(x: Array, z: Array) -> Array:
    """Orthogonal projection onto T_X St(n, r), compact form of (A.1).

    P_X(Z) = Z - X sym(X^T Z).
    """
    return z - x @ sym(x.T @ z)


def qr_retract(x: Array, xi: Array) -> Array:
    """QR retraction: thin QR of (X + ξ), Q with positive R-diagonal.

    Sign correction on diag(R) makes Retr_X(0) = X (unique thin QR).
    """
    q, r = jnp.linalg.qr(x + xi, mode="reduced")
    diag = jnp.diag(r)
    signs = jnp.where(diag >= 0, 1.0, -1.0)
    return q * signs


def cayley_omega(x: Array, xi: Array) -> Array:
    """Skew-symmetric generator Ω of the Cayley retraction (Appendix A.2.1).

    Ω̂ = ξ X^T - (1/2) X (X^T ξ X^T),  Ω = Ω̂ - Ω̂^T.

    For ξ ∈ T_X St this satisfies Ω X = ξ, so the Cayley curve has
    Y(0) = X and Y'(0) = ξ.
    """
    xt = x.T
    omega_hat = xi @ xt - 0.5 * x @ (xt @ xi @ xt)
    return omega_hat - omega_hat.T


def cayley_retract(
    x: Array,
    xi: Array,
    *,
    alpha: float = 1.0,
    n_iters: int = 8,
) -> Array:
    """Cayley retraction via the fixed-point iteration (A.3).

    Y_{k+1} = X + (α/2) Ω (X + Y_k),  Y_0 = (I + α Ω) X.

    Default α = 1 is the unit-step retraction of ξ (ξ already includes -η
    in Algorithm 1). A few iterations suffice (Appendix A.2.2).
    """
    omega = cayley_omega(x, xi)
    y = x + alpha * (omega @ x)
    half = 0.5 * alpha

    def body(_, y_k):
        return x + half * (omega @ (x + y_k))

    return jax.lax.fori_loop(0, n_iters, body, y)


def random_stiefel(key: Array, n: int, r: int, dtype=jnp.float32) -> Array:
    """Random point of St(n, r) via QR of a Gaussian matrix."""
    z = jax.random.normal(key, (n, r), dtype=dtype)
    q, r_mat = jnp.linalg.qr(z, mode="reduced")
    diag = jnp.diag(r_mat)
    signs = jnp.where(diag >= 0, 1.0, -1.0)
    return q * signs


def orth_error(x: Array) -> Array:
    """max |X^T X - I_r| (column-orthonormal)."""
    r = x.shape[1]
    return jnp.max(jnp.abs(x.T @ x - jnp.eye(r, dtype=x.dtype)))


def row_orth_error(a: Array) -> Array:
    """max |A A^T - I_r| (row-orthonormal factor A)."""
    r = a.shape[0]
    return jnp.max(jnp.abs(a @ a.T - jnp.eye(r, dtype=a.dtype)))
