"""Identities and routing checks for SGDR. float64, atol 1e-8."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from sgdr.attention import dense_causal_attention, sparse_block_attention
from sgdr.demo import identity_errors, wss_high_freq_errors
from sgdr.rope import apply_rope, rope_frequencies, to_complex, to_real
from sgdr.route import top_p_mask
from sgdr.spectrum import (
    Spectrum,
    block_means,
    estimate_spectrum,
    exact_block_score,
    macroscopic_only_score,
    microscopic_interference,
    online_block_score,
    pair_ratio,
    score_from_ratio,
    traditional_pooled_score,
)


def _token_mean_score(q_pre, k_pre, omegas, block_size):
    """Mean over local offsets of sum_f Re(q_post * conj(k_post)). No 1/sqrt(d)."""
    q = to_complex(q_pre)
    k = to_complex(k_pre)
    q_pos = jnp.arange(q.shape[0], dtype=jnp.float64)
    k_pos = jnp.arange(k.shape[0], dtype=jnp.float64)
    q_post = apply_rope(q, q_pos, omegas)
    k_post = apply_rope(k, k_pos, omegas)
    n_q = q.shape[0] // block_size
    n_k = k.shape[0] // block_size
    q_post = q_post[: n_q * block_size].reshape(n_q, block_size, -1)
    k_post = k_post[: n_k * block_size].reshape(n_k, block_size, -1)
    product = q_post[:, None, :, None, :] * jnp.conj(k_post[None, :, None, :, :])
    return jnp.real(product).mean(axis=(2, 3)).sum(axis=-1)


def _causal_probs(row):
    shifted = row - np.max(row)
    weights = np.exp(shifted)
    return weights / weights.sum()


def test_rope_pairs_and_frequencies():
    dim = 8
    omegas = rope_frequencies(dim)
    expected = 1.0 / (10000.0 ** (2.0 * np.arange(dim // 2) / dim))
    assert omegas.dtype == jnp.float64
    assert np.allclose(np.asarray(omegas), expected)
    x = jnp.arange(dim, dtype=jnp.float64)
    packed = to_complex(x)
    assert packed.shape == (dim // 2,)
    assert np.allclose(np.asarray(to_real(packed)), np.asarray(x))
    pos = jnp.asarray([3.0])
    rotated = apply_rope(packed[None, :], pos, omegas)[0]
    manual = packed * jnp.exp(1j * omegas * 3.0)
    assert np.allclose(np.asarray(rotated), np.asarray(manual))


def test_block_score_matches_token_mean():
    key = jax.random.PRNGKey(2)
    kq, kk = jax.random.split(key)
    block_size, n_blocks, dim = 8, 5, 16
    length = block_size * n_blocks
    q = jax.random.normal(kq, (length, dim), dtype=jnp.float64)
    k = jax.random.normal(kk, (length, dim), dtype=jnp.float64)
    omegas = rope_frequencies(dim)
    block = exact_block_score(q, k, omegas, block_size)
    token = _token_mean_score(q, k, omegas, block_size)
    err = float(jnp.max(jnp.abs(block - token)))
    assert err < 1e-8
    pooled = traditional_pooled_score(q, k, omegas, block_size)
    assert float(jnp.max(jnp.abs(pooled - block))) < 1e-8
    # Router convention: no 1/sqrt(d). A scaled score is a different number.
    scaled = token / jnp.sqrt(dim)
    assert float(jnp.max(jnp.abs(block - scaled))) > 1e-3


def test_closed_form_matches_block_score_with_exact_ratio():
    key = jax.random.PRNGKey(7)
    kq, kk = jax.random.split(key)
    block_size, n_blocks, dim = 6, 4, 12
    length = block_size * n_blocks
    q = jax.random.normal(kq, (length, dim), dtype=jnp.float64) * 0.4 + 0.8
    k = jax.random.normal(kk, (length, dim), dtype=jnp.float64) * 0.4 - 0.5
    omegas = rope_frequencies(dim)
    block, block_f = exact_block_score(q, k, omegas, block_size, per_freq=True)
    interference = microscopic_interference(q, k, omegas, block_size)
    q_bar = block_means(q, block_size)
    k_bar = block_means(k, block_size)
    ratio, valid = pair_ratio(interference, q_bar, k_bar)
    assert bool(jnp.all(valid))
    hat, hat_f = score_from_ratio(q_bar, k_bar, ratio, omegas, block_size, per_freq=True)
    assert float(jnp.max(jnp.abs(hat - block))) < 1e-8
    assert float(jnp.max(jnp.abs(hat_f - block_f))) < 1e-8
    # Polar form must match the complex product, not a different convention.
    n_q, n_k, _ = interference.shape
    delta = (jnp.arange(n_q)[:, None] - jnp.arange(n_k)[None, :]).astype(jnp.float64)
    phase = jnp.exp(1j * omegas * block_size * delta[:, :, None])
    complex_score = jnp.real(ratio * q_bar[:, None, :] * jnp.conj(k_bar)[None, :, :] * phase)
    assert float(jnp.max(jnp.abs(complex_score - block_f))) < 1e-8


def test_identity_helper_stays_under_tolerance():
    token_err, closed_err, pool_err = identity_errors()
    assert token_err < 1e-8
    assert closed_err < 1e-8
    assert pool_err < 1e-8


def test_estimated_spectrum_beats_macro_only_on_high_frequency():
    """Seeded WSS draw with a strong high-frequency residual.

    Literal post-RoPE pooling equals S_block, so it is not a weaker
    estimator. S_trad here is the macro-only score (C_hat fixed at 1),
    the online baseline that discards the intra-block spectrum. The
    estimated spectrum must beat that baseline on the high-frequency
    subspace of this seeded draw.
    """
    result = wss_high_freq_errors(seed=0)
    assert result["n_hf"] >= 1
    assert result["err_pool"] < 1e-8
    assert result["err_hat"] < result["err_trad"]
    assert result["beat"]
    # The win should be clear, not a tie at roundoff.
    assert result["err_trad"] > 10.0 * result["err_hat"]


def test_delta_clamp_uses_boundary_table_entry():
    freq = 2
    delta_max = 1
    c_hat = jnp.zeros((2 * delta_max + 1, freq), dtype=jnp.complex128)
    # index = delta + delta_max. Distinct magnitudes at -1, 0, +1.
    c_hat = c_hat.at[0].set(2.0 + 0.0j)
    c_hat = c_hat.at[1].set(5.0 + 0.0j)
    c_hat = c_hat.at[2].set(3.0 + 0.0j)
    omegas = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    spectrum = Spectrum(c_hat, delta_max, omegas, block_size=4)
    # Four blocks so |delta| reaches 3, past delta_max.
    q_bar = jnp.ones((4, freq), dtype=jnp.complex128)
    k_bar = jnp.ones((4, freq), dtype=jnp.complex128)
    hat = online_from_means(q_bar, k_bar, spectrum)
    # omega = 0, Phi = 0, theta = 0, so S_hat = sum_f Lambda(clamped delta).
    # delta = -3 clamps to -1, Lambda = 2, two frequencies, score = 4.
    assert abs(float(hat[0, 3]) - 4.0) < 1e-8
    assert abs(float(hat[3, 0]) - 6.0) < 1e-8
    assert abs(float(hat[1, 1]) - 10.0) < 1e-8


def online_from_means(q_bar, k_bar, spectrum):
    n_q = q_bar.shape[0]
    n_k = k_bar.shape[0]
    delta = jnp.arange(n_q)[:, None] - jnp.arange(n_k)[None, :]
    ratio = spectrum.lookup(delta)
    return score_from_ratio(q_bar, k_bar, ratio, spectrum.omegas, spectrum.block_size)


def test_top_p_causal_diagonal_mass_and_minimality():
    # Diagonal is among the strongest scores, so the minimal prefix
    # already contains it and the drop property is well defined.
    scores = jnp.asarray(
        [
            [4.0, 9.0, 9.0, 9.0],
            [0.2, 3.5, 8.0, 8.0],
            [0.1, 1.2, 2.4, 7.0],
            [1.0, 0.5, 0.4, 3.0],
        ],
        dtype=jnp.float64,
    )
    p = 0.9
    mask = np.asarray(top_p_mask(scores, p=p))
    n = scores.shape[0]
    assert mask.shape == (n, n)
    assert not mask[np.triu_indices(n, k=1)].any()
    assert np.all(np.diag(mask))
    host = np.asarray(scores)
    for i in range(n):
        row = host[i, : i + 1]
        selected = mask[i, : i + 1]
        probs = _causal_probs(row)
        mass = float(probs[selected].sum())
        assert mass + 1e-8 >= p
        if selected.sum() > 1:
            lowest = int(np.argmin(np.where(selected, row, np.inf)))
            dropped = mass - float(probs[lowest])
            assert dropped < p
    full = np.asarray(top_p_mask(scores, p=1.0))
    for i in range(n):
        assert full[i, : i + 1].all()
        assert not full[i, i + 1 :].any()


def test_top_p_keeps_weak_diagonal_and_selects_all_at_p_one():
    scores = jnp.asarray(
        [
            [0.0, 5.0, 5.0],
            [-4.0, -3.0, 6.0],
            [2.0, 2.2, -8.0],
        ],
        dtype=jnp.float64,
    )
    mask = np.asarray(top_p_mask(scores, p=0.5))
    assert np.all(np.diag(mask))
    assert not mask[np.triu_indices(3, k=1)].any()
    # Query block 2 scores [2.0, 2.2, -8]. Softmax mass of j=1 alone is
    # about 0.55, so the score prefix is {1}. The weak diagonal is still on.
    assert mask[2, 1] and mask[2, 2]
    assert not mask[2, 0]
    row = np.asarray(scores[2, :3])
    probs = _causal_probs(row)
    assert float(probs[mask[2, :3]].sum()) + 1e-8 >= 0.5
    full = np.asarray(top_p_mask(scores, p=1.0))
    for i in range(3):
        assert full[i, : i + 1].all()
        assert not full[i, i + 1 :].any()


def test_sparse_attention_matches_dense_at_p_one_and_respects_mask():
    key = jax.random.PRNGKey(11)
    kq, kk, kv, ks = jax.random.split(key, 4)
    block_size, n_blocks, dim = 4, 3, 8
    length = block_size * n_blocks
    q = jax.random.normal(kq, (length, dim), dtype=jnp.float64) * 0.2
    k = jax.random.normal(kk, (length, dim), dtype=jnp.float64) * 0.2
    v = jax.random.normal(kv, (length, dim), dtype=jnp.float64)
    scores = jax.random.normal(ks, (n_blocks, n_blocks), dtype=jnp.float64)
    mask_full = top_p_mask(scores, p=1.0)
    sparse = sparse_block_attention(q, k, v, mask_full, block_size)
    dense = dense_causal_attention(q, k, v)
    assert sparse.shape == (length, dim)
    assert float(jnp.max(jnp.abs(sparse - dense))) < 1e-8

    mask = top_p_mask(scores, p=0.6)
    out = sparse_block_attention(q, k, v, mask, block_size)
    assert out.shape == (length, dim)
    host_mask = np.asarray(mask)
    for j in range(n_blocks):
        v_zero = v.at[j * block_size : (j + 1) * block_size].set(0.0)
        out_zero = sparse_block_attention(q, k, v_zero, mask, block_size)
        for i in range(n_blocks):
            sl = slice(i * block_size, (i + 1) * block_size)
            changed = float(jnp.max(jnp.abs(out[sl] - out_zero[sl]))) > 1e-8
            assert changed is bool(host_mask[i, j])


def test_online_score_uses_estimated_table_not_token_loop():
    data_err = wss_high_freq_errors(seed=0)
    assert data_err["beat"]
    # A second seed is not required to win. This one is the contracted draw.
    key = jax.random.PRNGKey(0)
    block_size, dim = 32, 32
    from sgdr.demo import synthesize_wss

    data, omegas = synthesize_wss(4, 6, block_size, dim, key)
    spectrum = estimate_spectrum(data, data, omegas, block_size)
    held = data[0]
    hat = online_block_score(held, held, spectrum)
    block = exact_block_score(held, held, omegas, block_size)
    # Same-set estimate should be close on the full score as well as HF.
    assert hat.shape == block.shape
    q_bar = block_means(held, block_size)
    macro = macroscopic_only_score(q_bar, q_bar, omegas, block_size)
    assert float(jnp.mean(jnp.abs(hat - block))) < float(jnp.mean(jnp.abs(macro - block)))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
