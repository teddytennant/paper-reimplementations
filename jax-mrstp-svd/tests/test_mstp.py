"""MSTP-SVD / MRSTP-SVD checks from Theorems 4.2–4.3 and 5.1."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from jax_mrstp_svd.decompose import (
    mrstp_svd,
    mstp_reconstruct,
    mstp_svd,
    relative_frobenius_error,
    stp_svd_matrix,
)
from jax_mrstp_svd.product import (
    frobenius_norm,
    identity_tensor,
    matrix_stp,
    t_product,
    tensor_transpose,
)
from jax_mrstp_svd.rearrange import rearrange
from jax_mrstp_svd.transforms import get_transform


def _rand(key, shape):
    return random.normal(key, shape, dtype=jnp.float64)


def _planted_slice(key, m1, m2, n1, n2, k, scales):
    keys = random.split(key, 2 * k)
    A = jnp.zeros((m1 * m2, n1 * n2), dtype=jnp.float64)
    for i in range(k):
        B = _rand(keys[2 * i], (m1, n1))
        C = _rand(keys[2 * i + 1], (m2, n2))
        B = B / (frobenius_norm(B) + 1e-15)
        C = C / (frobenius_norm(C) + 1e-15)
        A = A + float(scales[i]) * jnp.kron(B, C)
    return A


def _planted_tensor(key, m1, m2, n1, n2, n3, k, transform="identity"):
    L = get_transform(transform)
    keys = random.split(key, n3)
    slices = []
    for j in range(n3):
        scales = [4.0 / (i + 1) for i in range(k)]
        slices.append(_planted_slice(keys[j], m1, m2, n1, n2, k, scales))
    A_bar = jnp.stack(slices, axis=2)
    return L.inverse(A_bar)


def test_matrix_stp_svd_orthogonal_and_error_decreases():
    """Theorem 4.2: U_i orthogonal; ||E_k||_F decreases in k."""
    m1, m2, n1, n2 = 3, 2, 3, 2
    A = _planted_slice(random.PRNGKey(0), m1, m2, n1, n2, k=3, scales=(5.0, 2.0, 0.7))
    recs = []
    for k in (1, 2, 3):
        factors = stp_svd_matrix(A, m1, m2, n1, n2, k=k)
        rec = sum(
            matrix_stp(matrix_stp(U, S), V.T) for U, S, V in factors
        )
        recs.append(rec)
        U0, _, V0 = factors[0]
        assert jnp.allclose(U0.T @ U0, jnp.eye(m1), atol=1e-10)
        assert jnp.allclose(V0.T @ V0, jnp.eye(n1), atol=1e-10)
    e1 = frobenius_norm(A - recs[0])
    e2 = frobenius_norm(A - recs[1])
    e3 = frobenius_norm(A - recs[2])
    assert e2 <= e1 + 1e-12
    assert e3 <= e2 + 1e-12
    assert e3 <= 1e-10


def test_mstp_orthogonal_factors_transform_domain():
    """Theorem 4.3 / Def. 2.7: U^⊤ *_L U = I (and likewise for V)."""
    m1, m2, n1, n2, n3 = 3, 2, 3, 2, 4
    A = _planted_tensor(random.PRNGKey(1), m1, m2, n1, n2, n3, k=2, transform="identity")
    factors = mstp_svd(A, m1, m2, n1, n2, k=1, transform="identity")
    fac = factors[0]
    Iu = identity_tensor(m1, n3, "identity")
    Iv = identity_tensor(n1, n3, "identity")
    UtU = t_product(tensor_transpose(fac.U, "identity"), fac.U, "identity")
    UUt = t_product(fac.U, tensor_transpose(fac.U, "identity"), "identity")
    VtV = t_product(tensor_transpose(fac.V, "identity"), fac.V, "identity")
    assert jnp.allclose(UtU, Iu, atol=1e-9)
    assert jnp.allclose(UUt, Iu, atol=1e-9)
    assert jnp.allclose(VtV, Iv, atol=1e-9)


def test_mstp_orthogonal_factors_fft():
    m1, m2, n1, n2, n3 = 2, 2, 2, 2, 3
    A = _planted_tensor(random.PRNGKey(2), m1, m2, n1, n2, n3, k=1, transform="fft")
    A = jnp.real(A)
    factors = mstp_svd(A, m1, m2, n1, n2, k=1, transform="fft")
    fac = factors[0]
    Iu = identity_tensor(m1, n3, "fft")
    UtU = t_product(tensor_transpose(fac.U, "fft"), fac.U, "fft")
    assert jnp.allclose(UtU, Iu, rtol=1e-8, atol=1e-8)


def test_multiterm_error_leq_single_term():
    """Paper motivation / Theorem 4.3: more terms cannot increase ||E||_F."""
    m1, m2, n1, n2, n3 = 3, 2, 3, 2, 3
    A = _rand(random.PRNGKey(3), (m1 * m2, n1 * n2, n3))
    f1 = mstp_svd(A, m1, m2, n1, n2, k=1, transform="identity")
    f2 = mstp_svd(A, m1, m2, n1, n2, k=2, transform="identity")
    f3 = mstp_svd(A, m1, m2, n1, n2, k=3, transform="identity")
    e1 = relative_frobenius_error(A, mstp_reconstruct(f1, "identity"))
    e2 = relative_frobenius_error(A, mstp_reconstruct(f2, "identity"))
    e3 = relative_frobenius_error(A, mstp_reconstruct(f3, "identity"))
    assert float(e2) <= float(e1) + 1e-12
    assert float(e3) <= float(e2) + 1e-12
    assert float(e1) > float(e3)


def test_planted_low_rank_exact_with_enough_terms():
    m1, m2, n1, n2, n3, k = 3, 2, 3, 2, 4, 2
    A = _planted_tensor(
        random.PRNGKey(4), m1, m2, n1, n2, n3, k=k, transform="identity"
    )
    factors = mstp_svd(A, m1, m2, n1, n2, k=k, transform="identity")
    rec = mstp_reconstruct(factors, "identity")
    assert float(relative_frobenius_error(A, rec)) < 1e-10


def test_mstp_error_matches_trailing_singular_values():
    """Theorem 4.3 eq. (8) with identity transform (ρ = 1)."""
    m1, m2, n1, n2, n3, k = 2, 2, 2, 2, 3, 1
    A = _rand(random.PRNGKey(5), (m1 * m2, n1 * n2, n3))
    factors = mstp_svd(A, m1, m2, n1, n2, k=k, transform="identity")
    rec = mstp_reconstruct(factors, "identity")
    err2 = float(frobenius_norm(A - rec) ** 2)
    predicted = 0.0
    v = min(m1 * n1, m2 * n2)
    for j in range(n3):
        R = rearrange(A[:, :, j], m1, m2, n1, n2)
        s = jnp.linalg.svd(R, compute_uv=False)
        predicted += float(jnp.sum(s[k:] ** 2))
    assert abs(err2 - predicted) / (predicted + 1e-15) < 1e-8
    assert v >= k


def test_mrstp_finite_and_small_on_planted():
    """Theorem 5.1: randomized reconstruction is finite; small on planted data."""
    m1, m2, n1, n2, n3, k = 3, 2, 3, 2, 3, 2
    A = _planted_tensor(
        random.PRNGKey(6), m1, m2, n1, n2, n3, k=k, transform="identity"
    )
    factors = mrstp_svd(
        A,
        m1,
        m2,
        n1,
        n2,
        k=k,
        transform="identity",
        oversampling=2,
        n_power=2,
        key=random.PRNGKey(99),
    )
    rec = mstp_reconstruct(factors, "identity")
    rel = float(relative_frobenius_error(A, rec))
    assert jnp.isfinite(rel)
    assert rel < 1e-6


def test_mrstp_fft_smoke():
    m1, m2, n1, n2, n3 = 2, 2, 2, 2, 4
    A = _rand(random.PRNGKey(7), (m1 * m2, n1 * n2, n3))
    factors = mrstp_svd(
        A,
        m1,
        m2,
        n1,
        n2,
        k=1,
        transform="fft",
        oversampling=2,
        n_power=1,
        key=random.PRNGKey(1),
    )
    rec = mstp_reconstruct(factors, "fft")
    rel = float(relative_frobenius_error(A, rec))
    assert jnp.isfinite(rel)
    # k=1 on a generic 4×4×4 tensor is a coarse sketch; only require a finite residual.
    assert rel < 2.0


def test_truncated_mstp_worse_or_equal_full():
    """Remark 4.3 / eq. (6): truncating B_i cannot improve the residual."""
    m1, m2, n1, n2, n3 = 3, 2, 3, 2, 2
    A = _rand(random.PRNGKey(8), (m1 * m2, n1 * n2, n3))
    full = mstp_reconstruct(
        mstp_svd(A, m1, m2, n1, n2, k=2, transform="identity"), "identity"
    )
    trunc = mstp_reconstruct(
        mstp_svd(A, m1, m2, n1, n2, k=2, transform="identity", rank=1),
        "identity",
    )
    e_full = float(frobenius_norm(A - full))
    e_trunc = float(frobenius_norm(A - trunc))
    assert e_full <= e_trunc + 1e-10
