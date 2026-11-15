"""Exact block scores, offline spectrum, and the closed-form online score.

Router scores do not include 1/sqrt(d). Equation 4 / Appendix B.1 define
S_block as the mean of token-level Re(q_post * conj(k_post)), summed over
frequency. The optional attention module applies 1/sqrt(d) separately.

Literal post-RoPE mean pooling equals S_block. The double sum of a product
q(n) * conj(k(m)) factors into the product of sums, so

    S_trad(i, j) = sum_f Re(mean_n q_post,f * conj(mean_m k_post,f))

is an identity, not an approximation. The baseline that actually drops
intra-block high-frequency phase is the macro-only score, C_hat = 1.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from sgdr.rope import apply_rope, to_complex


@dataclass
class Spectrum:
    """Offline structural spectrum indexed by relative block distance.

    C_hat[delta + delta_max, f] estimates I_f / (qbar_f * conj(kbar_f))
    for pairs with i - j = delta. Lambda and theta are the polar factors.
    """

    C_hat: jnp.ndarray
    delta_max: int
    omegas: jnp.ndarray
    block_size: int

    @property
    def lambda_amp(self):
        return jnp.abs(self.C_hat)

    @property
    def theta(self):
        return jnp.angle(self.C_hat)

    def lookup(self, delta):
        """Return C_hat at clamped delta. delta is an integer array."""
        clamped = jnp.clip(delta, -self.delta_max, self.delta_max)
        index = clamped + self.delta_max
        return self.C_hat[index]


def _full_blocks(length, block_size):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    n_blocks = int(length) // int(block_size)
    if n_blocks < 1:
        raise ValueError("sequence shorter than one block")
    return n_blocks


def block_means(x_pre, block_size):
    """Pre-RoPE complex block means, shape (n_blocks, F)."""
    x = to_complex(x_pre)
    n_blocks = _full_blocks(x.shape[0], block_size)
    blocked = x[: n_blocks * block_size].reshape(n_blocks, block_size, -1)
    return jnp.mean(blocked, axis=1)


def microscopic_interference(q_pre, k_pre, omegas, block_size):
    """I_f(i, j) from Appendix B.1, shape (n_q, n_k, F) complex.

    I_f = (1/B**2) * sum_{n,m} q_f(iB+n) * conj(k_f(jB+m)) * exp(i*omega_f*(n-m))

    The sum factors as Qrot_f(i) * conj(Krot_f(j)) / B**2, which is exact.
    """
    q = to_complex(q_pre)
    k = to_complex(k_pre)
    omegas = jnp.asarray(omegas, dtype=jnp.float64)
    n_q = _full_blocks(q.shape[0], block_size)
    n_k = _full_blocks(k.shape[0], block_size)
    bsz = int(block_size)
    q = q[: n_q * bsz].reshape(n_q, bsz, -1)
    k = k[: n_k * bsz].reshape(n_k, bsz, -1)
    offsets = jnp.arange(bsz, dtype=jnp.float64)
    rot = jnp.exp(1j * offsets[:, None] * omegas[None, :])
    q_rot = jnp.sum(q * rot[None, :, :], axis=1)
    k_rot = jnp.sum(k * rot[None, :, :], axis=1)
    return q_rot[:, None, :] * jnp.conj(k_rot)[None, :, :] / (bsz * bsz)


def _macro_phase(n_q, n_k, omegas, block_size):
    delta = jnp.arange(n_q)[:, None] - jnp.arange(n_k)[None, :]
    delta = delta.astype(jnp.float64)
    phase = jnp.exp(1j * omegas[None, None, :] * block_size * delta[:, :, None])
    return delta, phase


def exact_block_score(q_pre, k_pre, omegas, block_size, per_freq=False):
    """S_block(i, j) = sum_f Re(I_f * exp(i * omega_f * B * (i - j))).

    No 1/sqrt(d). If per_freq, also return S_f with shape (n_q, n_k, F).
    """
    interference = microscopic_interference(q_pre, k_pre, omegas, block_size)
    n_q, n_k, _ = interference.shape
    _, phase = _macro_phase(n_q, n_k, omegas, block_size)
    per_frequency = jnp.real(interference * phase)
    score = jnp.sum(per_frequency, axis=-1)
    if per_freq:
        return score, per_frequency
    return score


def traditional_pooled_score(q_pre, k_pre, omegas, block_size, per_freq=False):
    """Post-RoPE pooled baseline.

    S_trad(i, j) = sum_f Re(mean_n q_post,f * conj(mean_m k_post,f)).

    This equals S_block exactly. It is kept so the identity can be checked.
    """
    q = to_complex(q_pre)
    k = to_complex(k_pre)
    n_q = _full_blocks(q.shape[0], block_size)
    n_k = _full_blocks(k.shape[0], block_size)
    bsz = int(block_size)
    q_post = apply_rope(q, jnp.arange(q.shape[0], dtype=jnp.float64), omegas)
    k_post = apply_rope(k, jnp.arange(k.shape[0], dtype=jnp.float64), omegas)
    q_bar = q_post[: n_q * bsz].reshape(n_q, bsz, -1).mean(axis=1)
    k_bar = k_post[: n_k * bsz].reshape(n_k, bsz, -1).mean(axis=1)
    per_frequency = jnp.real(q_bar[:, None, :] * jnp.conj(k_bar)[None, :, :])
    score = jnp.sum(per_frequency, axis=-1)
    if per_freq:
        return score, per_frequency
    return score


def pair_ratio(interference, q_bar, k_bar, eps=1e-8):
    """Per-pair C_f = I_f / (qbar_f * conj(kbar_f)). Invalid pairs are 0."""
    denom = q_bar[:, None, :] * jnp.conj(k_bar)[None, :, :]
    valid = jnp.abs(denom) >= eps
    safe = jnp.where(valid, denom, 1.0)
    return jnp.where(valid, interference / safe, 0.0), valid


def score_from_ratio(q_bar, k_bar, ratio, omegas, block_size, per_freq=False):
    """Closed form (eq 12 / 39) from a per-pair or broadcast ratio.

    ratio: complex (n_q, n_k, F), or any shape that broadcasts to that.
    S_hat = sum_f Lambda * |qbar| * |kbar| * cos(omega*B*(i-j) + theta + Phi)
    with Phi = angle(qbar) - angle(kbar). No 1/sqrt(d).
    """
    q_bar = jnp.asarray(q_bar)
    k_bar = jnp.asarray(k_bar)
    ratio = jnp.asarray(ratio)
    n_q = q_bar.shape[0]
    n_k = k_bar.shape[0]
    delta, _ = _macro_phase(n_q, n_k, omegas, block_size)
    lam = jnp.abs(ratio)
    theta = jnp.angle(ratio)
    amp_q = jnp.abs(q_bar)[:, None, :]
    amp_k = jnp.abs(k_bar)[None, :, :]
    phi = jnp.angle(q_bar)[:, None, :] - jnp.angle(k_bar)[None, :, :]
    cos_arg = omegas[None, None, :] * block_size * delta[:, :, None] + theta + phi
    per_frequency = lam * amp_q * amp_k * jnp.cos(cos_arg)
    score = jnp.sum(per_frequency, axis=-1)
    if per_freq:
        return score, per_frequency
    return score


def macroscopic_only_score(q_bar, k_bar, omegas, block_size, per_freq=False):
    """Online score with the structural spectrum fixed at 1.

    This is the macro-only baseline: pre-RoPE means, macro RoPE phase,
    and no intra-block interference prior. High-frequency channels where
    the within-block kernel is far from 1 are where this baseline fails.
    """
    n_q = q_bar.shape[0]
    n_k = k_bar.shape[0]
    freq = q_bar.shape[-1]
    ratio = jnp.ones((n_q, n_k, freq), dtype=jnp.complex128)
    return score_from_ratio(q_bar, k_bar, ratio, omegas, block_size, per_freq=per_freq)


def _as_batch(x):
    x = jnp.asarray(x)
    if x.ndim == 2:
        return x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"expected (L, d) or (N, L, d), got {x.shape}")
    return x


def estimate_spectrum(q_pre, k_pre, omegas, block_size, eps=1e-8):
    """Mean per-delta ratio I_f / (qbar * conj(kbar)) over a calibration corpus.

    q_pre, k_pre: (L, d) or (N, L, d), pre-RoPE. Pairs whose denominator
    magnitude is below eps are skipped. delta_max is the largest |i-j|
    present in calibration. Missing deltas stay at 0.
    """
    q_pre = _as_batch(q_pre)
    k_pre = _as_batch(k_pre)
    if q_pre.shape[0] != k_pre.shape[0]:
        raise ValueError("q and k corpus sizes differ")
    omegas = jnp.asarray(omegas, dtype=jnp.float64)
    bsz = int(block_size)
    n_blocks = _full_blocks(q_pre.shape[1], bsz)
    if _full_blocks(k_pre.shape[1], bsz) != n_blocks:
        raise ValueError("q and k must cover the same number of blocks")
    freq = q_pre.shape[-1] // 2
    delta_max = n_blocks - 1
    n_delta = 2 * delta_max + 1
    sum_ratio = jnp.zeros((n_delta, freq), dtype=jnp.complex128)
    counts = jnp.zeros((n_delta, freq), dtype=jnp.float64)
    for seq in range(q_pre.shape[0]):
        interference = microscopic_interference(q_pre[seq], k_pre[seq], omegas, bsz)
        q_bar = block_means(q_pre[seq], bsz)
        k_bar = block_means(k_pre[seq], bsz)
        ratio, valid = pair_ratio(interference, q_bar, k_bar, eps=eps)
        delta = jnp.arange(n_blocks)[:, None] - jnp.arange(n_blocks)[None, :]
        index = (delta + delta_max).reshape(-1)
        sum_ratio = sum_ratio.at[index].add(ratio.reshape(-1, freq))
        counts = counts.at[index].add(valid.reshape(-1, freq).astype(jnp.float64))
    safe_counts = jnp.maximum(counts, 1.0)
    c_hat = jnp.where(counts > 0, sum_ratio / safe_counts, 0.0)
    return Spectrum(c_hat, delta_max, omegas, bsz)


def online_block_score(q_pre, k_pre, spectrum, per_freq=False):
    """Eq 12 from pre-RoPE tokens and an offline Spectrum.

    Lambda and theta are looked up at clamp(i - j, -delta_max, delta_max).
    The analytical macro phase omega_f * B * (i - j) uses the true distance.
    """
    q_bar = block_means(q_pre, spectrum.block_size)
    k_bar = block_means(k_pre, spectrum.block_size)
    n_q = q_bar.shape[0]
    n_k = k_bar.shape[0]
    delta = jnp.arange(n_q)[:, None] - jnp.arange(n_k)[None, :]
    ratio = spectrum.lookup(delta)
    return score_from_ratio(
        q_bar,
        k_bar,
        ratio,
        spectrum.omegas,
        spectrum.block_size,
        per_freq=per_freq,
    )


def within_block_kernel(omegas, block_size):
    """|mean_n exp(i * omega_f * n)|, the mean-phase kernel inside a block."""
    offsets = jnp.arange(int(block_size), dtype=jnp.float64)
    kernel = jnp.mean(jnp.exp(1j * offsets[:, None] * omegas[None, :]), axis=0)
    return jnp.abs(kernel)


def high_frequency_mask(omegas, block_size, kernel_tol=0.05):
    """Channels whose within-block RoPE kernel is below kernel_tol.

    Falls back to the single smallest kernel if none pass the tolerance.
    """
    kernel = within_block_kernel(omegas, block_size)
    mask = kernel < kernel_tol
    if not bool(jnp.any(mask)):
        mask = jnp.zeros_like(kernel, dtype=bool).at[jnp.argmin(kernel)].set(True)
    return mask
