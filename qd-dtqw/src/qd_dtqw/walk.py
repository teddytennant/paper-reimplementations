"""Discrete-time quantum walk on a ring (arXiv:2609.31070 §II.B).

Each intensity lives on a cycle of K = 2^{Nq} nodes. The state is
|ψ_pos⟩ ⊗ |ψ_coin⟩, initially |i0⟩ ⊗ |0⟩ (eq. 5). One noiseless step is
U = S (I_pos ⊗ H_coin) with S the coin-controlled ring shift, implemented
as a QFT incrementer (Fig. 1): Hadamard coin, QFT on position, diagonal
R_k = diag(1, exp(2πi/2^k)) phases (controlled on the coin), inverse QFT.

Qubit order is |q_{Nq-1} … q_0, coin⟩ with q_{Nq-1} the position MSB, so
the statevector index is pos * 2 + coin.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from qd_dtqw.noise import amplitude_damping, depolarize, ket_to_rho

_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
_I2 = np.eye(2, dtype=np.complex128)
_P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


def _R(k: int) -> np.ndarray:
    """Phase gate R_k = diag(1, exp(2πi / 2^k)) from Fig. 1."""
    return np.diag([1.0, np.exp(2.0j * np.pi / 2**k)]).astype(np.complex128)


def _krons(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _single(op: np.ndarray, qubit: int, n: int) -> np.ndarray:
    mats = [_I2] * n
    mats[qubit] = np.asarray(op, dtype=np.complex128)
    return _krons(mats)


def _controlled(op: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    mats0 = [_I2] * n
    mats0[control] = _P0
    mats1 = [_I2] * n
    mats1[control] = _P1
    mats1[target] = np.asarray(op, dtype=np.complex128)
    return _krons(mats0) + _krons(mats1)


def _swap(q1: int, q2: int, n: int) -> np.ndarray:
    if q1 == q2:
        return np.eye(2**n, dtype=np.complex128)
    d = 2**n
    U = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
        bits[q1], bits[q2] = bits[q2], bits[q1]
        j = 0
        for b in bits:
            j = (j << 1) | b
        U[j, i] = 1.0
    return U


def qft_position(n_pos: int, n_qubits: int) -> np.ndarray:
    """Standard QFT on qubits 0..n_pos-1 (MSB first), identity on the rest.

    H on each position qubit, then controlled-R_k from less-significant
    qubits, then bit-reversal SWAPs so F_{jk} = exp(2πi jk / K) / √K.
    """
    U = np.eye(2**n_qubits, dtype=np.complex128)
    for i in range(n_pos):
        U = _single(_H, i, n_qubits) @ U
        for j in range(i + 1, n_pos):
            k = j - i + 1
            U = _controlled(_R(k), control=j, target=i, n=n_qubits) @ U
    for i in range(n_pos // 2):
        U = _swap(i, n_pos - 1 - i, n_qubits) @ U
    return U


def omega_position(n_pos: int) -> np.ndarray:
    """Ω = R_1 ⊗ … ⊗ R_{Nq} on the position register (Razzoli eq. 12)."""
    return _krons([_R(k) for k in range(1, n_pos + 1)])


def shift_unitary(n_pos: int) -> np.ndarray:
    """Coin-controlled ring shift S via QFT incrementer (Fig. 1).

    Coin |0⟩ decrements position (mod K), coin |1⟩ increments.
    S = (F† ⊗ I) (Ω† ⊗ |0⟩⟨0| + Ω ⊗ |1⟩⟨1|) (F ⊗ I).
    """
    n_qubits = n_pos + 1
    F = qft_position(n_pos, n_qubits)
    om = omega_position(n_pos)
    coin_omega = np.kron(om.conj().T, _P0) + np.kron(om, _P1)
    return F.conj().T @ coin_omega @ F


def coin_unitary(n_pos: int) -> np.ndarray:
    """C = I_pos ⊗ H (Hadamard coin of Fig. 1)."""
    return _single(_H, n_pos, n_pos + 1)


@lru_cache(maxsize=8)
def one_step_unitary(n_pos: int) -> np.ndarray:
    """One DTQW step U = S (I ⊗ H_coin), as a (2^{Nq+1} × 2^{Nq+1}) matrix."""
    return shift_unitary(n_pos) @ coin_unitary(n_pos)


def explicit_shift_unitary(n_pos: int) -> np.ndarray:
    """Circulant shift used only to check the QFT incrementer."""
    K = 2**n_pos
    dim = 2 * K
    S = np.zeros((dim, dim), dtype=np.complex128)
    for pos in range(K):
        S[(pos - 1) % K * 2 + 0, pos * 2 + 0] = 1.0
        S[(pos + 1) % K * 2 + 1, pos * 2 + 1] = 1.0
    return S


def basis_state(n_pos: int, position: int, coin: int = 0) -> np.ndarray:
    """Computational-basis ket |position⟩ ⊗ |coin⟩ (eq. 5 uses coin=0)."""
    K = 2**n_pos
    if not (0 <= int(position) < K):
        raise ValueError(f"position {position} not in [0, {K})")
    if coin not in (0, 1):
        raise ValueError("coin must be 0 or 1")
    psi = np.zeros(2 * K, dtype=np.complex128)
    psi[int(position) * 2 + int(coin)] = 1.0
    return psi


def position_projectors(n_pos: int) -> np.ndarray:
    """P_i = |i⟩⟨i| ⊗ I_coin (eq. 8), stacked as (K, dim, dim)."""
    K = 2**n_pos
    dim = 2 * K
    Ps = np.zeros((K, dim, dim), dtype=np.complex128)
    for i in range(K):
        Ps[i, i * 2, i * 2] = 1.0
        Ps[i, i * 2 + 1, i * 2 + 1] = 1.0
    return Ps


def position_probs(state: np.ndarray | jnp.ndarray, n_pos: int) -> jnp.ndarray:
    """p(i) = Tr(P_i ρ) with P_i = |i⟩⟨i| ⊗ I, or the pure-state analogue."""
    K = 2**n_pos
    arr = np.asarray(state, dtype=np.complex128)
    if arr.ndim == 1:
        amp = arr.reshape(K, 2)
        p = np.sum(np.abs(amp) ** 2, axis=1)
    elif arr.ndim == 2:
        p = np.zeros(K, dtype=np.float64)
        for i in range(K):
            p[i] = np.real(arr[i * 2, i * 2] + arr[i * 2 + 1, i * 2 + 1])
        p = np.maximum(p, 0.0)
    else:
        raise ValueError("state must be a ket or a density matrix")
    p = np.real(p)
    s = p.sum()
    if s > 0:
        p = p / s
    return jnp.asarray(p)


def _apply_channel(rho: np.ndarray, channel: str | None, p_noise: float) -> np.ndarray:
    if channel is None or p_noise <= 0.0:
        return rho
    n_qubits = int(np.log2(rho.shape[0]))
    if channel in ("depolarize", "depolarizing"):
        return depolarize(rho, p_noise)
    if channel in ("amplitude_damping", "ad", "ampdamp"):
        return amplitude_damping(rho, p_noise, n_qubits)
    raise ValueError(f"unknown channel {channel!r}")


class DTQW:
    """Classically simulated DTQW on a K=2^{n_pos}-cycle, one coin qubit."""

    def __init__(self, n_pos: int = 2):
        if n_pos < 1:
            raise ValueError("n_pos must be >= 1")
        self.n_pos = int(n_pos)
        self.K = 2**self.n_pos
        self.n_qubits = self.n_pos + 1
        self.dim = 2 * self.K
        self.U = one_step_unitary(self.n_pos)
        self.U_jax = jnp.asarray(self.U)
        self._projectors = position_projectors(self.n_pos)

    def initial_ket(self, i0: int) -> np.ndarray:
        return basis_state(self.n_pos, int(i0), coin=0)

    def initial_rho(self, i0: int) -> np.ndarray:
        return ket_to_rho(self.initial_ket(i0))

    def evolve_ket(self, i0: int, t: int) -> np.ndarray:
        """Noiseless |ψ_t⟩ = U^t |i0⟩⊗|0⟩ (eq. 6 as a pure state)."""
        psi = self.initial_ket(i0)
        if t <= 0:
            return psi
        return np.linalg.matrix_power(self.U, int(t)) @ psi

    def evolve_rho(
        self,
        i0: int,
        t: int,
        channel: str | None = None,
        p_noise: float = 0.0,
    ) -> np.ndarray:
        """ρ after t steps. channel=None is exact unitary eq. 6.

        With a channel, noise is applied after every U (NISQ stand-in for
        eq. 7, which the paper writes as E(U_t ρ0 U_t†) on hardware).
        """
        rho = self.initial_rho(i0)
        if t <= 0:
            return rho
        U = self.U
        Ud = U.conj().T
        for _ in range(int(t)):
            rho = U @ rho @ Ud
            rho = _apply_channel(rho, channel, p_noise)
        return rho

    def probs(
        self,
        i0: int,
        t: int,
        channel: str | None = None,
        p_noise: float = 0.0,
    ) -> jnp.ndarray:
        """Occupancy p(i_t) from eq. 8."""
        if channel is None or p_noise <= 0.0:
            return position_probs(self.evolve_ket(i0, t), self.n_pos)
        return position_probs(self.evolve_rho(i0, t, channel, p_noise), self.n_pos)

    def one_step_probs(self, i_t: int) -> jnp.ndarray:
        """Reinit |ψ_pos⟩=|i_t⟩, |ψ_coin⟩=|0⟩, one DTQW step (training pair)."""
        return self.probs(int(i_t), t=1, channel=None)

    def transition_matrix(self) -> jnp.ndarray:
        """Q_{ij} = q(x'=j | x=i) for one reinitialized step (eq. 4)."""
        rows = [np.asarray(self.one_step_probs(i), dtype=np.float64) for i in range(self.K)]
        return jnp.asarray(np.stack(rows, axis=0))

    def sample_position(self, probs, key) -> jnp.ndarray:
        p = jnp.asarray(probs, dtype=jnp.float64)
        p = p / jnp.clip(p.sum(), 1e-12, None)
        return jax.random.categorical(key, jnp.log(jnp.clip(p, 1e-12, 1.0)))

    def training_pair(
        self,
        i0: int,
        t: int,
        key,
        channel: str | None = None,
        p_noise: float = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Measure x_t after t steps, reinit coin, one more step → x_{t+1}."""
        k1, k2 = jax.random.split(key)
        p_t = self.probs(i0, t, channel, p_noise)
        i_t = self.sample_position(p_t, k1)
        p_next = self.one_step_probs(int(i_t))
        i_next = self.sample_position(p_next, k2)
        return i_t, i_next

    def forward_image(
        self,
        x0: jnp.ndarray,
        t: int,
        key,
        channel: str | None = None,
        p_noise: float = 0.0,
    ) -> jnp.ndarray:
        """Independent DTQW on every pixel/channel/voxel of a discrete image."""
        x0 = jnp.asarray(x0)
        flat = np.asarray(x0, dtype=np.int32).reshape(-1)
        n = flat.size
        keys = jax.random.split(key, n)
        out = np.empty(n, dtype=np.int32)
        for idx, (i0, k) in enumerate(zip(flat, keys)):
            p = self.probs(int(i0), t, channel, p_noise)
            out[idx] = int(self.sample_position(p, k))
        return jnp.asarray(out.reshape(x0.shape))

    def training_pair_image(
        self,
        x0: jnp.ndarray,
        t: int,
        key,
        channel: str | None = None,
        p_noise: float = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per-pixel (x_t, x_{t+1}) with coin reinitialized at x_t."""
        x0 = jnp.asarray(x0)
        shape = x0.shape
        flat = np.asarray(x0, dtype=np.int32).reshape(-1)
        n = flat.size
        keys = jax.random.split(key, n)
        xt = np.empty(n, dtype=np.int32)
        xt1 = np.empty(n, dtype=np.int32)
        for idx, (i0, k) in enumerate(zip(flat, keys)):
            i_t, i_next = self.training_pair(int(i0), t, k, channel, p_noise)
            xt[idx] = int(i_t)
            xt1[idx] = int(i_next)
        return jnp.asarray(xt.reshape(shape)), jnp.asarray(xt1.reshape(shape))
