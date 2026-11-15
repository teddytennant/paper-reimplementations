"""Square orthonormal linear codecs, Section 2.2.

Columns of Phi are the basis vectors. Encode and decode are equation (4):
c = Phi.T @ y, y = Phi @ c. Columns are orthonormal, so Phi @ Phi.T = I.
"""

import jax.numpy as jnp

from mercerflow.common import default_float_dtype


def fix_signs(Phi):
    """Flip basis columns so the first row is non-negative.

    This is the sign convention stated for the Mercer basis in equation (5)
    and reused for PCA.
    """
    signs = jnp.where(Phi[0] < 0.0, -1.0, 1.0)
    return Phi * signs


def encode(Phi, y):
    """c = Phi.T @ y. y has shape (L,) or (batch, L)."""
    y = jnp.asarray(y)
    if y.ndim == 1:
        return Phi.T @ y
    if y.ndim == 2:
        return y @ Phi
    raise ValueError("y must have shape (L,) or (batch, L)")


def decode(Phi, c):
    """y = Phi @ c. c has shape (L,) or (batch, L)."""
    c = jnp.asarray(c)
    if c.ndim == 1:
        return Phi @ c
    if c.ndim == 2:
        return c @ Phi.T
    raise ValueError("c must have shape (L,) or (batch, L)")


def raw_basis(length, dtype=None):
    """Raw codec: Phi = I."""
    dtype = default_float_dtype() if dtype is None else dtype
    return jnp.eye(int(length), dtype=dtype)


def mercer_basis(K):
    """Mercer codec, equation (5).

    Columns of Phi are eigenvectors of K with eigenvalues in descending
    order, so K = Phi @ diag(eigenvalues) @ Phi.T. The first row of Phi
    is constrained to be non-negative.

    For a centred draw y ~ N(0, K), Cov(Phi.T @ y) = eigenvalues.
    """
    gram = 0.5 * (K + K.T)
    evals_asc, evecs = jnp.linalg.eigh(gram)
    eigenvalues = evals_asc[::-1]
    Phi = fix_signs(evecs[:, ::-1])
    return Phi, eigenvalues


def fourier_basis(length, dtype=None):
    """Real orthonormal DFT, equation (6).

    Column 0 is the DC mode L^{-1/2}. Frequencies m = 1, 2, ... contribute
    a cosine column then a sine column, each scaled by sqrt(2/L). When L
    is even, the final column is the Nyquist mode L^{-1/2} cos(pi s).
    Time index s runs over {0, ..., L-1}.
    """
    dtype = default_float_dtype() if dtype is None else dtype
    length = int(length)
    if length < 1:
        raise ValueError("length must be positive")
    s = jnp.arange(length, dtype=dtype)
    inv_sqrt = jnp.asarray(length, dtype=dtype) ** -0.5
    pair_scale = jnp.sqrt(jnp.asarray(2.0, dtype=dtype) / length)
    columns = [jnp.full((length,), inv_sqrt, dtype=dtype)]
    n_pairs = (length - 1) // 2
    two_pi = jnp.asarray(2.0, dtype=dtype) * jnp.pi
    for harmonic in range(1, n_pairs + 1):
        angle = two_pi * harmonic * s / length
        columns.append(pair_scale * jnp.cos(angle))
        columns.append(pair_scale * jnp.sin(angle))
    if length % 2 == 0:
        nyquist = inv_sqrt * jnp.cos(jnp.asarray(jnp.pi, dtype=dtype) * s)
        columns.append(nyquist)
    return jnp.stack(columns, axis=1).astype(dtype)


def dct_basis(length, dtype=None):
    """Orthonormal type-II DCT, equation (7).

    phi_0(s) = L^{-1/2}
    phi_m(s) = sqrt(2/L) cos(pi m (s + 1/2) / L) for m = 1, ..., L-1.
    Columns are ordered from lowest frequency to highest.
    """
    dtype = default_float_dtype() if dtype is None else dtype
    length = int(length)
    if length < 1:
        raise ValueError("length must be positive")
    s = jnp.arange(length, dtype=dtype)
    mode = jnp.arange(length, dtype=dtype)
    argument = jnp.pi * mode[None, :] * (s[:, None] + 0.5) / length
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=dtype) / length)
    Phi = scale * jnp.cos(argument)
    inv_sqrt = jnp.asarray(length, dtype=dtype) ** -0.5
    Phi = Phi.at[:, 0].set(inv_sqrt)
    return Phi.astype(dtype)


def pca_basis(windows):
    """PCA codec, equation (8).

    Eigenvectors of the training-window sample covariance
    Sigma_hat = (1/S) sum_k (y^{(k)} - y_bar) (y^{(k)} - y_bar)^T,
    eigenvalues descending, first row of Phi non-negative.

    Centering is used only to build Sigma_hat. encode does not subtract
    the training mean.
    """
    windows = jnp.asarray(windows)
    if windows.ndim != 2:
        raise ValueError("windows must have shape (S, L)")
    n_windows = windows.shape[0]
    if n_windows < 1:
        raise ValueError("need at least one training window")
    mean = jnp.mean(windows, axis=0)
    centered = windows - mean
    covariance = (centered.T @ centered) / n_windows
    covariance = 0.5 * (covariance + covariance.T)
    evals_asc, evecs = jnp.linalg.eigh(covariance)
    eigenvalues = evals_asc[::-1]
    Phi = fix_signs(evecs[:, ::-1])
    return Phi, eigenvalues


def build_basis(name, length, K=None, windows=None, dtype=None):
    """Build one of the five codecs by name. Returns Phi only."""
    key = name.lower()
    if key == "raw":
        return raw_basis(length, dtype=dtype)
    if key == "fourier":
        return fourier_basis(length, dtype=dtype)
    if key == "dct":
        return dct_basis(length, dtype=dtype)
    if key == "mercer":
        if K is None:
            raise ValueError("Mercer codec requires K")
        Phi, _ = mercer_basis(K)
        return Phi
    if key == "pca":
        if windows is None:
            raise ValueError("PCA codec requires training windows")
        Phi, _ = pca_basis(windows)
        return Phi
    raise ValueError(f"unknown codec {name}")
