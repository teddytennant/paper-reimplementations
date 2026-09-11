"""Lemma 15 and Eqs. (6)–(9): heavy-coordinate truncation of D."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_dwmf.truncation import (
    frobenius_norm_diag,
    heavy_set,
    sample_count_R,
    sample_index,
    truncate_diagonal,
)


def test_heavy_set_picks_large_diagonal_eq6():
    # Eq. (6): H_ϵ = {i ∈ [n] : d_i ≥ ϵ}
    d = jnp.asarray([1.0, 0.5, 0.09, 0.2, 0.0])
    eps = 0.1
    H = np.asarray(heavy_set(d, eps))
    assert set(H.tolist()) == {0, 1, 3}


def test_heavy_set_cardinality_bound_eq7():
    # Eq. (7): |H_ϵ| ≤ ||D||_F^2 / ϵ^2, since Σ d_i^2 = ||D||_F^2 and
    # each heavy coordinate has d_i^2 ≥ ϵ^2.
    d = jnp.asarray([3.0, 2.0, 0.5, 0.4, 0.01, 0.0])
    eps = 0.3
    H = heavy_set(d, eps)
    fro2 = float(frobenius_norm_diag(d) ** 2)
    assert len(H) <= fro2 / (eps**2) + 1e-12
    assert len(H) == 4  # 3, 2, 0.5, 0.4


def test_sample_count_matches_lemma15_eq8():
    # Eq. (8) / Lemma 15: R = ceil( (||D||_F^2 / ϵ^2) ln(||D||_F^2 / (ϵ^2 δ)) )
    # Main text writes "log"; the Lemma 15 proof uses e^{-R ϵ^2 / ||D||_F^2},
    # so the logarithm is natural.
    fro, eps, delta = 2.0, 0.25, 1e-3
    R = sample_count_R(fro, eps, delta)
    ratio = (fro**2) / (eps**2)
    expected = int(np.ceil(ratio * np.log(ratio / delta)))
    assert R == expected


def test_sq_sampling_probabilities():
    # Problem setting: SQ(D) samples i with probability d_i^2 / ||D||_F^2.
    d = jnp.asarray([2.0, 1.0, 0.0])
    p_true = np.square(np.asarray(d))
    p_true = p_true / p_true.sum()
    key = jax.random.PRNGKey(0)
    n_draw = 4000
    keys = jax.random.split(key, n_draw)
    draws = np.array([int(sample_index(keys[r], d)) for r in range(n_draw)])
    emp = np.bincount(draws, minlength=3) / n_draw
    np.testing.assert_allclose(emp, p_true, atol=0.04)


def test_truncate_diagonal_op_norm_eq9_exact_event():
    # Eq. (9) / Lemma 15 successful event: ||D − D̃||_op ≤ ϵ and
    # s = rank(D̃) ≤ ||D||_F^2 / ϵ^2. Light coordinates are never kept.
    d = jnp.asarray([1.2, 0.8, 0.05, 0.04, 0.0])
    eps = 0.1
    key = jax.random.PRNGKey(1)
    S, d_S = truncate_diagonal(key, d, eps, delta=1e-6, exact=True)
    d_tilde = np.zeros(d.shape[0])
    d_tilde[np.asarray(S)] = np.asarray(d_S)
    op = float(np.max(np.abs(np.asarray(d) - d_tilde)))  # diagonal ⇒ ||·||_op
    assert op <= eps + 1e-15
    fro = float(frobenius_norm_diag(d))
    assert S.shape[0] <= (fro / eps) ** 2 + 1e-12
    assert set(np.asarray(S).tolist()) == {0, 1}
    assert not any(int(i) in set(np.asarray(S).tolist()) for i in (2, 3, 4))


def test_sampled_truncation_recovers_heavy_indices():
    # Lemma 15: R independent SQ(D) samples recover H_ϵ with probability ≥ 1−δ.
    d = jnp.asarray([2.0, 1.5, 0.01, 0.0])
    eps = 0.2
    delta = 1e-3
    key = jax.random.PRNGKey(2)
    S, _ = truncate_diagonal(key, d, eps, delta, exact=False)
    H = set(np.asarray(heavy_set(d, eps)).tolist())
    assert H.issubset(set(np.asarray(S).tolist()))


def test_truncation_zero_or_large_eps_returns_empty():
    # Lemma 15: if ||D||_F = 0 or ϵ ≥ ||D||_F, D̃ = 0 already satisfies the bound.
    d = jnp.asarray([0.3, 0.4])  # ||D||_F = 0.5
    S, d_S = truncate_diagonal(jax.random.PRNGKey(0), d, eps=0.5, delta=0.1)
    assert S.shape[0] == 0
    assert d_S.shape[0] == 0
    S0, _ = truncate_diagonal(
        jax.random.PRNGKey(0), jnp.zeros(4), eps=0.1, delta=0.1
    )
    assert S0.shape[0] == 0
