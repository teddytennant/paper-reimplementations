"""Critical-grid DFT, transfer matrices A_U, and the finite-support norms of Theorem 3."""

from __future__ import annotations

import jax.numpy as jnp

from muon_c.polar import polar_svd


def rms(x: jnp.ndarray) -> jnp.ndarray:
    """RMS(O) = ||O||_F / sqrt(|O|) as in Section 5.4."""
    return jnp.linalg.norm(x) / jnp.sqrt(x.size)


def critical_fft(kernel: jnp.ndarray) -> jnp.ndarray:
    """Orthonormal 2D DFT along the spatial kernel axes (kh, kw).

    jax.numpy.fft.fft2 with norm='ortho' is the paper's F_{Omega_k}, so
    Uhat(p, q) = A_U(theta_p, phi_q) / sqrt(n) with n = kh * kw (eq. after (2)).
    """
    return jnp.fft.fft2(kernel, axes=(-2, -1), norm="ortho")


def critical_ifft(freq: jnp.ndarray) -> jnp.ndarray:
    """Inverse orthonormal 2D DFT along the last two axes."""
    return jnp.fft.ifft2(freq, axes=(-2, -1), norm="ortho")


def transfer_matrix(U: jnp.ndarray, theta1, theta2) -> jnp.ndarray:
    """Unnormalized channel-transfer matrix A_U(theta) from eq. (1).

    A_U(theta) = sum_{u,v} U_{uv} exp(-i (u theta1 + v theta2)), shape Cout x Cin.
    """
    kh, kw = U.shape[-2], U.shape[-1]
    u = jnp.arange(kh)
    v = jnp.arange(kw)
    phase = jnp.exp(-1j * (u[:, None] * theta1 + v[None, :] * theta2))
    return jnp.einsum("...uv,uv->...", U.astype(jnp.complex64), phase)


def transfer_on_grid(U: jnp.ndarray, n_grid: int = 32) -> jnp.ndarray:
    """A_U on an n_grid x n_grid frequency lattice, shape (Cout, Cin, G, G)."""
    kh, kw = U.shape[-2], U.shape[-1]
    th = 2.0 * jnp.pi * jnp.arange(n_grid) / n_grid
    u = jnp.arange(kh)
    v = jnp.arange(kw)
    phase_h = jnp.exp(-1j * u[None, :] * th[:, None])
    phase_w = jnp.exp(-1j * v[None, :] * th[:, None])
    Uc = U.astype(jnp.complex64)
    return jnp.einsum("ijuv,pu,qv->ijpq", Uc, phase_h, phase_w)


def _opnorm(blocks: jnp.ndarray) -> jnp.ndarray:
    """Largest singular value of each matrix in a batch (last two axes)."""
    s = jnp.linalg.svd(blocks, compute_uv=False)
    return s[..., 0]


def spatial_norm(U: jnp.ndarray) -> jnp.ndarray:
    """||U||_sp = max_{u,v} ||U_uv||_op (Theorem 3)."""
    blocks = jnp.moveaxis(U, (-4, -3), (-2, -1))
    return jnp.max(_opnorm(blocks.astype(jnp.float32)))


def patch_norm(U: jnp.ndarray) -> jnp.ndarray:
    """||U||_patch = ||B_U||_op, B_U the Cout x (Cin kh kw) unfolding."""
    cout, cin, kh, kw = U.shape
    B = U.reshape(cout, cin * kh * kw)
    return _opnorm(B.astype(jnp.float32))


def sampled_norm(U: jnp.ndarray) -> jnp.ndarray:
    """||U||_samp = max_{p,q} ||A_U(theta_p, phi_q)||_op = sqrt(n) max ||Uhat||_op."""
    n = U.shape[-2] * U.shape[-1]
    Uhat = critical_fft(U)
    blocks = jnp.moveaxis(Uhat, (-4, -3), (-2, -1))
    return jnp.sqrt(n) * jnp.max(_opnorm(blocks))


def conv_norm_grid(U: jnp.ndarray, n_grid: int = 32) -> jnp.ndarray:
    """Grid estimate of ||U||_conv = sup_theta ||A_U(theta)||_op (paper: conv,32)."""
    A = transfer_on_grid(U, n_grid=n_grid)
    blocks = jnp.moveaxis(A, (-4, -3), (-2, -1))
    return jnp.max(_opnorm(blocks))


def lebesgue_constant(k: int, n_theta: int = 4096) -> jnp.ndarray:
    """One-dimensional Lebesgue constant Lambda_k of Theorem 3.

    ell_p^{(k)}(theta) = (1/k) sum_{r=0}^{k-1} exp(-i r (theta - 2 pi p / k)),
    Lambda_k = sup_theta sum_p |ell_p^{(k)}(theta)|.
    """
    thetas = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    r = jnp.arange(k)
    p = jnp.arange(k)
    phase = thetas[None, :] - 2.0 * jnp.pi * p[:, None] / k
    ell = jnp.sum(jnp.exp(-1j * r[:, None, None] * phase[None, :, :]), axis=0) / k
    return jnp.max(jnp.sum(jnp.abs(ell), axis=0))


def unfold(kernel: jnp.ndarray) -> jnp.ndarray:
    """Cout x (Cin * kh * kw) patch unfolding B_W of Section 4.1."""
    cout, cin, kh, kw = kernel.shape[-4:]
    return kernel.reshape(*kernel.shape[:-4], cout, cin * kh * kw)


def spatial_polar_direction(M: jnp.ndarray) -> jnp.ndarray:
    """Muon-S: polarize each Cout x Cin spatial-offset block independently."""
    blocks = jnp.moveaxis(M, (-4, -3), (-2, -1))
    out = polar_svd(blocks)
    return jnp.moveaxis(out, (-2, -1), (-4, -3))
