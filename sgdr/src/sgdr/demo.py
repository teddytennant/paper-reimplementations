"""Print exact-identity errors and the seeded WSS comparison.

Literal post-RoPE pooling equals S_block, so it cannot lose as an estimator
of the mean logit. S_trad in the seeded print is the macro-only score
(structural spectrum fixed at 1), which drops the intra-block phase the
offline spectrum is meant to restore. The residual is synthetic.
"""

import jax
import jax.numpy as jnp

from sgdr.rope import rope_frequencies, to_complex
from sgdr.spectrum import (
    block_means,
    estimate_spectrum,
    exact_block_score,
    high_frequency_mask,
    macroscopic_only_score,
    microscopic_interference,
    online_block_score,
    pair_ratio,
    score_from_ratio,
    traditional_pooled_score,
)


def _reference_token_mean(q_pre, k_pre, omegas, block_size):
    """Independent mean of Re(q_post * conj(k_post)) over local offsets."""
    q = to_complex(q_pre)
    k = to_complex(k_pre)
    q_post = q * jnp.exp(1j * jnp.arange(q.shape[0])[:, None] * omegas[None, :])
    k_post = k * jnp.exp(1j * jnp.arange(k.shape[0])[:, None] * omegas[None, :])
    n_q = q.shape[0] // block_size
    n_k = k.shape[0] // block_size
    q_post = q_post[: n_q * block_size].reshape(n_q, block_size, -1)
    k_post = k_post[: n_k * block_size].reshape(n_k, block_size, -1)
    product = q_post[:, None, :, None, :] * jnp.conj(k_post)[None, :, None, :, :]
    return jnp.real(product).mean(axis=(2, 3)).sum(axis=-1)


def identity_errors(seed=0, n_blocks=4, block_size=8, dim=16):
    """Return (token-mean error, closed-form error, literal pool error)."""
    key = jax.random.PRNGKey(seed)
    kq, kk = jax.random.split(key)
    length = n_blocks * block_size
    q = jax.random.normal(kq, (length, dim), dtype=jnp.float64) * 0.5 + 0.7
    k = jax.random.normal(kk, (length, dim), dtype=jnp.float64) * 0.5 - 0.4
    omegas = rope_frequencies(dim)
    block = exact_block_score(q, k, omegas, block_size)
    token = _reference_token_mean(q, k, omegas, block_size)
    pooled = traditional_pooled_score(q, k, omegas, block_size)
    interference = microscopic_interference(q, k, omegas, block_size)
    q_bar = block_means(q, block_size)
    k_bar = block_means(k, block_size)
    ratio, _ = pair_ratio(interference, q_bar, k_bar)
    closed = score_from_ratio(q_bar, k_bar, ratio, omegas, block_size)
    return (
        float(jnp.max(jnp.abs(block - token))),
        float(jnp.max(jnp.abs(closed - block))),
        float(jnp.max(jnp.abs(pooled - block))),
    )


def synthesize_wss(
    n_seq,
    n_blocks,
    block_size,
    dim,
    key,
    amp=3.0,
    noise=0.02,
    mean_scale=0.5,
):
    """Shared complex mean plus a harmonic residual at frequency -omega_f.

    The residual is the same law in every block. Its lag covariance is
    amp^2 * exp(-i * omega_f * lag), plus white noise of variance noise^2
    on each complex channel. High-frequency RoPE channels have a near-zero
    within-block mean kernel, so the residual dominates I_f while the
    pre-RoPE block mean stays near the shared semantic mean.
    """
    omegas = rope_frequencies(dim)
    freq = dim // 2
    length = n_blocks * block_size
    k_mu_r, k_mu_i, k_nr, k_ni = jax.random.split(key, 4)
    mu = jax.random.normal(k_mu_r, (freq,), dtype=jnp.float64) + 1j * jax.random.normal(
        k_mu_i, (freq,), dtype=jnp.float64
    )
    mu = mu / jnp.abs(mu) * mean_scale
    time = jnp.arange(length, dtype=jnp.float64)
    phase = jnp.exp(-1j * time[:, None] * omegas[None, :])
    eps = noise * (
        jax.random.normal(k_nr, (n_seq, length, freq), dtype=jnp.float64)
        + 1j * jax.random.normal(k_ni, (n_seq, length, freq), dtype=jnp.float64)
    )
    hidden = mu[None, None, :] + amp * phase[None, :, :] + eps
    real = jnp.stack([jnp.real(hidden), jnp.imag(hidden)], axis=-1)
    return real.reshape(n_seq, length, dim), omegas


def wss_high_freq_errors(
    seed=0,
    n_seq=8,
    n_blocks=6,
    block_size=44,
    dim=32,
    amp=3.0,
    noise=0.01,
):
    """Compare S_hat to S_block and to the macro-only baseline on HF channels.

    The returned err_trad is the macro-only error. Literal post-RoPE pooling
    is also reported and should sit at roundoff.
    """
    data, omegas = synthesize_wss(
        n_seq,
        n_blocks,
        block_size,
        dim,
        jax.random.PRNGKey(seed),
        amp=amp,
        noise=noise,
    )
    spectrum = estimate_spectrum(data[:-1], data[:-1], omegas, block_size)
    held = data[-1]
    _, block_f = exact_block_score(held, held, omegas, block_size, per_freq=True)
    _, hat_f = online_block_score(held, held, spectrum, per_freq=True)
    q_bar = block_means(held, block_size)
    _, macro_f = macroscopic_only_score(q_bar, q_bar, omegas, block_size, per_freq=True)
    _, pooled_f = traditional_pooled_score(held, held, omegas, block_size, per_freq=True)
    hf = high_frequency_mask(omegas, block_size)
    err_hat = float(jnp.mean(jnp.abs(hat_f[:, :, hf] - block_f[:, :, hf])))
    err_trad = float(jnp.mean(jnp.abs(macro_f[:, :, hf] - block_f[:, :, hf])))
    err_pool = float(jnp.max(jnp.abs(pooled_f - block_f)))
    return {
        "err_hat": err_hat,
        "err_trad": err_trad,
        "err_pool": err_pool,
        "beat": err_hat < err_trad,
        "n_hf": int(jnp.sum(hf)),
    }


def main():
    token_err, closed_err, pool_err = identity_errors()
    wss = wss_high_freq_errors()
    print(f"exact identity error: {token_err:.3e}")
    print(f"closed-form identity error: {closed_err:.3e}")
    print(f"literal post-RoPE pool identity error: {pool_err:.3e}")
    print(f"S_hat beat S_trad on seeded WSS: {wss['beat']}")


if __name__ == "__main__":
    main()
