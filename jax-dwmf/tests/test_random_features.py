"""Tiny optimized-random-feature sampler (Eqs. (1), (21)–(24), Lemma 20, E5)."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_dwmf.algorithm import materialize_N, operator_norm, true_f_M
from jax_dwmf.random_features import (
    cosine_kernel_entry,
    dft_matrix,
    estimate_kernel_entry,
    sample_optimized_features,
    shift_invariant_kernel,
)
from jax_dwmf.truncation import frobenius_norm_diag


def test_kernel_cosine_form_e5():
    # Eq. (E5): k(α, β) = G^{-1} Σ_y Q^{(τ)}(y) cos(2π y·(α−β)/G)
    G = 8
    Q = jnp.asarray([0.4, 0.2, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05])
    k = shift_invariant_kernel(Q)
    for a in range(G):
        for b in range(G):
            np.testing.assert_allclose(
                float(k[a, b]), float(cosine_kernel_entry(Q, a, b)), atol=1e-12
            )
    # Hermitian / real symmetric (paper: k is PSD real symmetric).
    np.testing.assert_allclose(np.asarray(k), np.asarray(k).T, atol=1e-12)
    evals = np.linalg.eigvalsh(np.asarray(k))
    assert evals.min() >= -1e-10


def test_kernel_matches_fourier_d3():
    # Eq. (D3) / (21): k = F† Q^{(τ)} F
    G = 6
    Q = jnp.linspace(0.5, 0.1, G)
    F = dft_matrix(G)
    k_fourier = np.asarray(jnp.conjugate(F).T @ jnp.diag(Q.astype(F.dtype)) @ F)
    k = np.asarray(shift_invariant_kernel(Q))
    np.testing.assert_allclose(k, k_fourier.real, atol=1e-10)


def test_lemma20_mc_close_on_small_grid():
    # Lemma 20 / Eq. (E8): Monte Carlo estimate of a kernel entry.
    G = 8
    Q = jnp.ones(G)
    exact = float(cosine_kernel_entry(Q, 0, 1))
    est = float(
        estimate_kernel_entry(
            jax.random.PRNGKey(0), Q, 0, 1, eps=0.05, delta=1e-3, Q_max=1.0
        )
    )
    # Hoeffding guarantee is high-probability ϵ-accuracy; on this tiny
    # instance we only check the estimator is in a few ϵ of the exact value.
    assert abs(est - exact) < 0.2


def test_frobenius_of_sqrt_qhat_is_one_eq26():
    # Eq. (26): ||sqrt(q̂^{(ρ)})||_F = 1 for a normalized distribution.
    q = jnp.asarray([0.5, 0.3, 0.2, 0.0])
    d = jnp.sqrt(q)
    np.testing.assert_allclose(float(frobenius_norm_diag(d)), 1.0, atol=1e-12)


def test_sigma_inv_principal_block_vs_dense():
    # Eqs. (1), (22), (D11): N ≈ Σ_ϵ^{-1} on a small 1-D grid.
    G = 6
    q = jnp.asarray([0.45, 0.25, 0.15, 0.1, 0.04, 0.01])
    q = q / jnp.sum(q)
    Q = jnp.asarray([1.0, 0.4, 0.2, 0.1, 0.05, 0.05])
    k = shift_invariant_kernel(Q)
    eps_reg = 0.3
    eps_D = 0.15  # keeps the two heaviest sqrt(q) coordinates
    samples, w, repr_N = sample_optimized_features(
        jax.random.PRNGKey(1),
        q,
        Q,
        eps_reg,
        eps_D=eps_D,
        n_samples=20,
        exact_support=True,
        k=k,
    )
    N = materialize_N(repr_N)
    d = jnp.sqrt(q)
    fM = true_f_M(d, k, eps_reg, lambda x: 1.0 / x)
    err = float(operator_norm(fM - N))
    # Inverse-specific bound (E22) with a = ϵ (A = k ⪰ 0 ⇒ Σ_ϵ ⪰ ϵ I).
    A_op = float(operator_norm(k))
    D_op = float(jnp.max(d))
    nu = 2.0 * A_op * D_op * eps_D
    assert err <= nu / (eps_reg**2) + 1e-9, (err, nu / (eps_reg**2))
    # Sampler returns indices in the grid.
    assert samples.shape == (20,)
    assert int(jnp.min(samples)) >= 0
    assert int(jnp.max(samples)) < G
    # Weights (E26) are finite; the two heaviest q-coordinates are in S.
    assert jnp.all(jnp.isfinite(w))
    S = set(np.asarray(repr_N.S).tolist())
    assert 0 in S and 1 in S


def test_sparsity_of_orf_representation():
    G = 5
    q = jnp.asarray([0.7, 0.2, 0.05, 0.04, 0.01])
    Q = jnp.ones(G)
    k = shift_invariant_kernel(Q)
    _, _, repr_N = sample_optimized_features(
        jax.random.PRNGKey(2),
        q,
        Q,
        eps_reg=0.4,
        eps_D=0.3,  # sqrt(0.7)≈0.84, sqrt(0.2)≈0.45, rest < 0.23
        n_samples=1,
        exact_support=True,
        k=k,
    )
    N = np.asarray(materialize_N(repr_N))
    S = set(np.asarray(repr_N.S).tolist())
    for i in range(G):
        for j in range(G):
            if not (i in S and j in S) and i != j:
                assert abs(N[i, j]) < 1e-14
