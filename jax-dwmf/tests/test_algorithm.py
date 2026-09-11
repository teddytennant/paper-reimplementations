"""Algorithm 1 / Theorem 16: principal-block matrix function and operator-norm error."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_dwmf.algorithm import (
    apply_f_eigh,
    diagonal_truncation_transform,
    materialize_N,
    operator_norm,
    perturbation_nu,
    query_N,
    true_f_M,
)
from jax_dwmf.truncation import frobenius_norm_diag, heavy_set


def _spd_pair(n=6, seed=0):
    """Small Hermitian A and nonnegative diagonal D."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    G = jax.random.normal(k1, (n, n))
    A = 0.5 * (G + G.T)
    # Mix of heavy and light diagonal weights.
    d = jnp.asarray([1.5, 1.2, 0.9, 0.04, 0.02, 0.0][:n], dtype=jnp.float64)
    if n > 6:
        extra = jax.random.uniform(k2, (n - 6,), minval=0.0, maxval=0.03)
        d = jnp.concatenate([d, extra])
    return d, A


def _f_inv(x):
    return 1.0 / x


def _f_exp(x):
    return jnp.exp(-x)


def test_block_eigh_matches_numpy_eq11():
    # Eq. (11) / Alg. 1 steps 4–5: B = f(M̃_S) by diagonalizing the block.
    rng = np.random.default_rng(0)
    H = rng.normal(size=(4, 4))
    block = 0.5 * (H + H.T) + np.eye(4)
    f = lambda x: 1.0 / x
    B = np.asarray(apply_f_eigh(jnp.asarray(block), f))
    evals, evecs = np.linalg.eigh(block)
    B_np = (evecs * f(evals)) @ evecs.T
    np.testing.assert_allclose(B, B_np, atol=1e-12)
    # JAX eigh of the same block, applied entrywise, must match.
    evals_j, evecs_j = jnp.linalg.eigh(jnp.asarray(block))
    B_j = np.asarray((evecs_j * f(evals_j)) @ evecs_j.T)
    np.testing.assert_allclose(B, B_j, atol=1e-12)


def test_principal_block_construction_eq10_c1():
    # Eq. (10) / (C1): M̃_S(t, u) = γ δ_{t,u} + d̃_{i_t} A(i_t, i_u) d̃_{i_u}
    d, A = _spd_pair(n=5, seed=3)
    gamma = 0.4
    eps_D = 0.1
    key = jax.random.PRNGKey(0)
    out = diagonal_truncation_transform(
        key, d, A, gamma, _f_inv, eps_D=eps_D, exact_support=True
    )
    S = np.asarray(out.S)
    dS = np.asarray(out.d_tilde)
    A_S = np.asarray(A)[np.ix_(S, S)]
    M_S = gamma * np.eye(len(S)) + (dS[:, None] * A_S * dS[None, :])
    B_direct = np.asarray(apply_f_eigh(jnp.asarray(M_S), _f_inv))
    np.testing.assert_allclose(np.asarray(out.B), B_direct, atol=1e-12)


def test_sparsity_pattern_eq12_c2():
    # Eq. (12) / (C2): N(i, j) = B(π(i), π(j)) on S×S, f(γ) on the
    # complement diagonal, and 0 otherwise.
    d, A = _spd_pair(n=6, seed=4)
    gamma = 0.5
    eps_D = 0.15
    out = diagonal_truncation_transform(
        jax.random.PRNGKey(1), d, A, gamma, _f_inv, eps_D=eps_D, exact_support=True
    )
    N = np.asarray(materialize_N(out))
    S = set(np.asarray(out.S).tolist())
    n = out.n
    f_gamma = out.f_gamma
    for i in range(n):
        for j in range(n):
            val = complex(query_N(out, i, j))
            if i in S and j in S:
                t, u = out.pi(i), out.pi(j)
                np.testing.assert_allclose(val, complex(out.B[t, u]), atol=1e-12)
            elif i == j:
                np.testing.assert_allclose(val, f_gamma, atol=1e-12)
            else:
                np.testing.assert_allclose(val, 0.0, atol=1e-12)
            np.testing.assert_allclose(N[i, j], val, atol=1e-12)
    # Off-block, off-diagonal entries of the materialized matrix vanish.
    for i in range(n):
        for j in range(n):
            if not (i in S and j in S) and i != j:
                assert abs(N[i, j]) < 1e-14


def test_row_supports_algorithm1_step6():
    # Alg. 1 step 6: J_t = {i_u ∈ S : B(t, π(i_u)) ≠ 0}
    d, A = _spd_pair(n=5, seed=5)
    out = diagonal_truncation_transform(
        jax.random.PRNGKey(2),
        d,
        A,
        gamma=0.3,
        f=_f_inv,
        eps_D=0.1,
        exact_support=True,
    )
    S_list = [int(i) for i in out.S]
    for t, Jt in enumerate(out.J):
        expected = tuple(
            S_list[u]
            for u in range(out.s)
            if abs(complex(out.B[t, u])) > 0.0
        )
        assert Jt == expected


def test_block_diagonal_form_eq13_c15():
    # Eq. (13) / (C15): after permuting S first, γI + D̃ A D̃ ≃ diag(M̃_S, γ I_{n-s}).
    d, A = _spd_pair(n=6, seed=6)
    gamma = 0.25
    eps_D = 0.1
    out = diagonal_truncation_transform(
        jax.random.PRNGKey(3), d, A, gamma, _f_inv, eps_D=eps_D, exact_support=True
    )
    S = np.asarray(out.S)
    complement = np.array([i for i in range(out.n) if i not in set(S.tolist())])
    perm = np.concatenate([S, complement])
    d_tilde = np.zeros(out.n)
    d_tilde[S] = np.asarray(out.d_tilde)
    M_tilde = gamma * np.eye(out.n) + np.diag(d_tilde) @ np.asarray(A) @ np.diag(d_tilde)
    M_perm = M_tilde[np.ix_(perm, perm)]
    s = out.s
    dS = np.asarray(out.d_tilde)
    A_S = np.asarray(A)[np.ix_(S, S)]
    M_S = gamma * np.eye(s) + dS[:, None] * A_S * dS[None, :]
    np.testing.assert_allclose(M_perm[:s, :s], M_S, atol=1e-12)
    np.testing.assert_allclose(M_perm[:s, s:], 0.0, atol=1e-12)
    np.testing.assert_allclose(M_perm[s:, :s], 0.0, atol=1e-12)
    np.testing.assert_allclose(M_perm[s:, s:], gamma * np.eye(out.n - s), atol=1e-12)


def test_operator_norm_error_below_theorem16_bound():
    """Theorem 16 / Eqs. (15), (19), (C4): ||f(M) − N||_op ≤ C(1+log(1+s)) L ν.

    On the successful event of Lemma 15 (exact_support=True) with exact
    queries (ϵ_A = 0), ν = 2 ||A||_op ||D||_op ϵ_D.

    Lemma 9's numerical constant C is not given a numerical value in the
    paper. For the inverse on spectra bounded away from 0 the paper itself
    replaces C(1+log(1+s))L by 1/a^2 (Eq. (E22)). We check both:

    * the inverse-specific bound ||X^{-1} − Y^{-1}||_op ≤ a^{-2} ||X − Y||_op
      with a = γ (spectra of M, M̃ sit in [γ, ∞) when A ⪰ 0; here A is
      shifted to be PSD);
    * a documented relaxation of the general Lipschitz bound using C = 1,
      which is the constant appearing in the scalar Lipschitz inequality.
    """
    n = 6
    key = jax.random.PRNGKey(7)
    G = jax.random.normal(key, (n, n))
    A = 0.5 * (G + G.T)
    # Make A PSD so M ⪰ γ I and the inverse Lipschitz constant is 1/γ^2.
    evals_A, evecs_A = jnp.linalg.eigh(A)
    A = (evecs_A * jnp.clip(evals_A, 0.0)) @ evecs_A.T
    d = jnp.asarray([1.4, 1.1, 0.8, 0.05, 0.02, 0.0])
    gamma = 0.5
    eps_D = 0.08
    f = _f_inv
    out = diagonal_truncation_transform(
        key, d, A, gamma, f, eps_D=eps_D, exact_support=True
    )
    N = materialize_N(out)
    fM = true_f_M(d, A, gamma, f)
    err = float(operator_norm(fM - N))

    A_op = float(operator_norm(A))
    D_op = float(jnp.max(d))
    D_F = float(frobenius_norm_diag(d))
    nu = perturbation_nu(A_op, D_op, D_F, eps_D, eps_A=0.0)
    s = out.s
    # Inverse-specific bound (E22) with a = γ: this is the bound the paper
    # uses for optimized random features, and it does not involve C or log s.
    inv_bound = nu / (gamma**2)
    assert err <= inv_bound + 1e-10, (err, inv_bound)

    # Scalar Lipschitz constant of x ↦ 1/x on [γ, ∞) is L = 1/γ^2.
    L = 1.0 / (gamma**2)
    # Documented relaxation of (C4): C = 1 (scalar Lipschitz, no log factor).
    # The paper's C(1+log(1+s)) is a worst-case operator-modulus factor from
    # Aleksandrov–Peller; on these n=6 instances the scalar bound already holds.
    relaxed = L * nu
    assert err <= relaxed + 1e-10, (err, relaxed)


def test_lipschitz_exp_operator_norm():
    # Same construction with f(x) = exp(−x), which is 1-Lipschitz on [0, ∞).
    n = 5
    key = jax.random.PRNGKey(8)
    G = jax.random.normal(key, (n, n))
    A = 0.5 * (G + G.T)
    d = jnp.asarray([1.0, 0.7, 0.03, 0.01, 0.0])
    gamma = 0.2
    eps_D = 0.05
    out = diagonal_truncation_transform(
        key, d, A, gamma, _f_exp, eps_D=eps_D, exact_support=True
    )
    N = materialize_N(out)
    fM = true_f_M(d, A, gamma, _f_exp)
    err = float(operator_norm(fM - N))
    A_op = float(operator_norm(A))
    D_op = float(jnp.max(d))
    D_F = float(frobenius_norm_diag(d))
    nu = perturbation_nu(A_op, D_op, D_F, eps_D, 0.0)
    L = 1.0  # |d/dx exp(−x)| = exp(−x) ≤ 1 on [0, ∞); spectra of M sit in R
    # Weyl: spectra of M and M̃ differ by at most ν, and exp is globally 1-Lipschitz.
    assert err <= L * nu + 1e-10, (err, L * nu)


def test_rank_bound_eq18():
    # Theorem 1 / Eq. (18): s ≤ ||D||_F^2 / ϵ_D^2
    d, A = _spd_pair(n=6, seed=9)
    eps_D = 0.1
    out = diagonal_truncation_transform(
        jax.random.PRNGKey(0),
        d,
        A,
        gamma=0.4,
        f=_f_inv,
        eps_D=eps_D,
        exact_support=True,
    )
    fro = float(frobenius_norm_diag(d))
    assert out.s <= (fro / eps_D) ** 2 + 1e-12
    assert set(np.asarray(out.S).tolist()) == set(
        np.asarray(heavy_set(d, eps_D)).tolist()
    )
