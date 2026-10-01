"""Activation-space orthogonal projectors and the raw-column penalty."""

import jax
import jax.numpy as jnp


def orthogonal_factor(V):
    """Thin-QR orthogonal factor of V. Columns are orthonormal.

    V has shape (d, k). A zero-column V returns shape (d, 0).
    """
    d, k = V.shape
    if k == 0:
        return jnp.zeros((d, 0), dtype=V.dtype)
    Q, _ = jnp.linalg.qr(V, mode="reduced")
    return Q


def sample_direction_mask(key, k, dropout_p):
    """Bernoulli(1 - p) mask in {0, 1}^k. No 1/(1-p) rescaling."""
    if dropout_p < 0.0 or dropout_p > 1.0:
        raise ValueError("dropout_p must lie in [0, 1]")
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    keep_prob = 1.0 - dropout_p
    return jax.random.bernoulli(key, keep_prob, (k,)).astype(dtype)


def warmup_alpha(step, warmup_steps):
    """Linear warm-up of alpha from 0 to 1. alpha stays in [0, 1]."""
    if warmup_steps <= 0:
        return 1.0
    return float(min(1.0, step / warmup_steps))


def project_activations(x, U, alpha=1.0, mask=None):
    """Apply P = I - alpha * U @ diag(m) @ U.T without forming P.

    x is (..., d) and U is (d, k). mask is m in {0, 1}^k, or None for ones.
    """
    k = U.shape[-1]
    if k == 0 or alpha == 0:
        return x
    coeff = x @ U
    if mask is not None:
        coeff = coeff * mask
    return x - alpha * (coeff @ U.T)


def dense_projector(U, alpha=1.0, mask=None):
    """Dense P = I - alpha * U @ diag(m) @ U.T. For checks, not the forward."""
    d, k = U.shape
    eye = jnp.eye(d, dtype=U.dtype)
    if k == 0 or alpha == 0:
        return eye
    if mask is None:
        middle = U @ U.T
    else:
        middle = (U * mask) @ U.T
    return eye - alpha * middle


def forward_input(W, x, b, U, alpha=1.0, mask=None):
    """Input-side projected linear map.

    y = W @ (x - alpha * U @ (m * (U.T @ x))) + b.
    W is (dout, din), U is (din, k). Bias is added and never projected.
    """
    x_proj = project_activations(x, U, alpha=alpha, mask=mask)
    return x_proj @ W.T + b


def forward_output(W, x, b, U, alpha=1.0, mask=None):
    """Output-side projected linear map.

    y = P @ (W @ x) + b, with U in R^{dout x k}.
    """
    pre = x @ W.T
    return project_activations(pre, U, alpha=alpha, mask=mask) + b


def orthogonality_penalty(V):
    """Mean off-diagonal inner product of the raw columns of V.

    L = 1/(k*(k-1)) * sum_{i != j} <v_i, v_j> for k > 1, else 0.
    Not a cosine and not a square. Columns are not normalized first.
    """
    k = V.shape[-1]
    if k <= 1:
        return jnp.zeros((), dtype=V.dtype)
    gram = V.T @ V
    off = jnp.sum(gram) - jnp.trace(gram)
    return off / (k * (k - 1))
