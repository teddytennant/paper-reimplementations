"""Lock DTQW forward-process equations of arXiv:2609.31070."""

from __future__ import annotations

import numpy as np

from qd_dtqw.walk import (
    DTQW,
    explicit_shift_unitary,
    one_step_unitary,
    position_probs,
    qft_position,
    shift_unitary,
)


N_POS = 2
K = 4


def test_one_step_is_unitary():
    """Noiseless circuit: U† U = I."""
    U = one_step_unitary(N_POS)
    ident = U.conj().T @ U
    np.testing.assert_allclose(ident, np.eye(U.shape[0]), atol=1e-10)
    ident_r = U @ U.conj().T
    np.testing.assert_allclose(ident_r, np.eye(U.shape[0]), atol=1e-10)


def test_qft_incrementer_matches_circulant_shift():
    """Fig. 1 shift equals the ring increment/decrement operator."""
    S = shift_unitary(N_POS)
    S_exp = explicit_shift_unitary(N_POS)
    np.testing.assert_allclose(S, S_exp, atol=1e-10)


def test_qft_matrix_is_dft():
    n_pos = N_POS
    F_full = qft_position(n_pos, n_pos + 1)
    K = 2**n_pos
    F = np.zeros((K, K), dtype=np.complex128)
    for j in range(K):
        for k in range(K):
            F[j, k] = np.exp(2j * np.pi * j * k / K) / np.sqrt(K)
    # F_full = F ⊗ I_coin, columns grouped as (pos, coin)
    for pos in range(K):
        for coin in range(2):
            col = F_full[:, pos * 2 + coin].reshape(K, 2)[:, coin]
            np.testing.assert_allclose(col, F[:, pos], atol=1e-10)


def test_probabilities_nonnegative_and_sum_to_one():
    """p(i) = Tr(P_i ρ) from eq. 8 is a categorical distribution."""
    walk = DTQW(N_POS)
    for i0 in range(K):
        for t in range(0, 6):
            p = np.asarray(walk.probs(i0, t))
            assert p.shape == (K,)
            assert np.all(p >= -1e-12)
            np.testing.assert_allclose(p.sum(), 1.0, atol=1e-10)


def test_eq8_trace_formula():
    walk = DTQW(N_POS)
    rho = walk.evolve_rho(i0=1, t=3)
    p = np.asarray(position_probs(rho, N_POS))
    traces = np.array(
        [np.real(np.trace(walk._projectors[i] @ rho)) for i in range(K)]
    )
    np.testing.assert_allclose(p, traces, atol=1e-10)


def test_ring_wraparound():
    """Walker at 0 reaches K-1 and walker at K-1 reaches 0 in one step."""
    walk = DTQW(N_POS)
    p0 = np.asarray(walk.probs(i0=0, t=1))
    # |0⟩⊗|0⟩ --H--> |0⟩⊗|+⟩ --S--> 1/√2 (|K-1⟩|0⟩ + |1⟩|1⟩)
    np.testing.assert_allclose(p0[(K - 1)], 0.5, atol=1e-10)
    np.testing.assert_allclose(p0[1], 0.5, atol=1e-10)
    np.testing.assert_allclose(p0[0], 0.0, atol=1e-10)

    p_last = np.asarray(walk.probs(i0=K - 1, t=1))
    np.testing.assert_allclose(p_last[K - 2], 0.5, atol=1e-10)
    np.testing.assert_allclose(p_last[0], 0.5, atol=1e-10)


def test_independent_pixels():
    """Two pixels with different i0 evolve independently (product occupancy)."""
    walk = DTQW(N_POS)
    t = 4
    p_a = np.asarray(walk.probs(i0=0, t=t))
    p_b = np.asarray(walk.probs(i0=3, t=t))
    assert not np.allclose(p_a, p_b, atol=1e-8)
    joint = np.outer(p_a, p_b)
    np.testing.assert_allclose(joint.sum(), 1.0, atol=1e-10)
    # A change in B's start does not change A's marginal.
    p_a2 = np.asarray(walk.probs(i0=0, t=t))
    np.testing.assert_allclose(p_a, p_a2, atol=1e-12)


def test_noiseless_does_not_converge_to_uniform():
    """Theoretical DTQW never converges to uniform (unitary is invertible)."""
    walk = DTQW(N_POS)
    uni = np.full(K, 1.0 / K)
    distances = []
    for t in range(1, 33):
        p = np.asarray(walk.probs(i0=0, t=t))
        distances.append(np.max(np.abs(p - uni)))
    assert min(distances) > 0.1
    # Periodicity on the 4-cycle: t=8 returns near the start, still peaked.
    p8 = np.asarray(walk.probs(i0=0, t=8))
    assert np.max(np.abs(p8 - uni)) > 0.2


def test_noise_closer_to_uniform_than_noiseless():
    """Depolarizing channel (eq. 7 stand-in) mixes toward the uniform prior."""
    walk = DTQW(N_POS)
    t = 16
    p_u = np.asarray(walk.probs(i0=0, t=t))
    p_n = np.asarray(walk.probs(i0=0, t=t, channel="depolarize", p_noise=0.35))
    uni = np.full(K, 1.0 / K)

    def kl(p):
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        return float(np.sum(p * np.log(p / uni)))

    assert kl(p_n) < kl(p_u)
    assert kl(p_n) < 0.05


def test_training_pair_reinitializes_coin():
    """From i_t, reinit |ψ_pos⟩=|i_t⟩, |ψ_coin⟩=|0⟩, then one DTQW step.

    That is not the same as continuing the unmeasured walk for t+1 steps:
    measuring x_t and resetting the coin (Fig. 1 / training pairs) yields a
    different x_{t+1} occupancy than U^{t+1}|i0⟩⊗|0⟩.
    """
    walk = DTQW(N_POS)
    i_t = 1
    p_reinit = np.asarray(walk.one_step_probs(i_t))
    p_from_ket = np.asarray(walk.probs(i_t, t=1))
    np.testing.assert_allclose(p_reinit, p_from_ket, atol=1e-12)

    psi = walk.initial_ket(i_t)
    assert abs(psi[i_t * 2 + 0] - 1.0) < 1e-12  # coin |0⟩
    assert abs(psi[i_t * 2 + 1]) < 1e-12

    t = 2
    p_t = np.asarray(walk.probs(0, t))
    mix = np.zeros(K)
    for i in range(K):
        mix += p_t[i] * np.asarray(walk.one_step_probs(i))
    p_cont = np.asarray(walk.probs(0, t + 1))
    assert np.max(np.abs(mix - p_cont)) > 0.2

    Q = np.asarray(walk.transition_matrix())
    np.testing.assert_allclose(Q[i_t], p_reinit, atol=1e-12)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-10)
