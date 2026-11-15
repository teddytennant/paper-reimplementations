"""CPU tests for EFQ-Softmax identities (paper eqs 8-11, Algorithms 1-2, II-D)."""

import jax
import jax.numpy as jnp
import numpy as np

from efq_softmax.attention import online_attention, softmax_attention
from efq_softmax.quantize import (
    C8,
    C16,
    H_BALANCE,
    H_MEAN,
    H_MMLU,
    TAU_BALANCE,
    TAU_MEAN,
    TAU_MMLU,
    affine_codes,
    apply_operand,
    block_scale_and_residual,
    efq_lut,
    efq_quantize,
    lut_index,
)

jax.config.update("jax_enable_x64", True)

C8_SET = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}


def _random_shifted(key, shape, lo=-8.0):
    x = jax.random.uniform(key, shape, minval=lo, maxval=0.0)
    return x - jnp.max(x, axis=-1, keepdims=True)


def test_codes_in_c8_and_reconstructed_values():
    key = jax.random.PRNGKey(0)
    x = _random_shifted(key, (4, 64))
    for tau, h in [(TAU_MMLU, H_MMLU), (TAU_MEAN, H_MEAN), (TAU_BALANCE, H_BALANCE)]:
        op = efq_quantize(x, tau=tau, h=h, block_size=32)
        codes = np.asarray(op.codes)
        assert codes.min() >= 0 and codes.max() <= 7
        q = np.asarray(C8[op.codes])
        assert set(np.unique(q)).issubset(C8_SET)
        _, _, _, s_b, _, _ = block_scale_and_residual(x, 32)
        q_b = C8[op.codes].reshape(*x.shape[:-1], s_b.shape[-1], 32)
        p_check = np.asarray((s_b[..., None] * q_b).reshape(*x.shape))
        np.testing.assert_allclose(np.asarray(op.ptilde), p_check, rtol=1e-12, atol=1e-12)


def test_scale_rule_eq8_eq9():
    key = jax.random.PRNGKey(1)
    x = _random_shifted(key, (8, 32), lo=-12.0)
    _, m_b, k_b, s_b, _, _ = block_scale_and_residual(x, 32)
    two_exp_over_9 = 2.0 * jnp.exp(m_b) / 9.0
    lo = jnp.exp2(k_b)
    hi = jnp.exp2(k_b + 1.0)
    np.testing.assert_allclose(np.asarray(s_b), np.asarray(lo), rtol=1e-12)
    assert np.all(np.asarray(lo) <= np.asarray(two_exp_over_9) + 1e-12)
    assert np.all(np.asarray(two_exp_over_9) < np.asarray(hi) + 1e-12)


def test_residual_identity():
    key = jax.random.PRNGKey(2)
    x = _random_shifted(key, (3, 5, 32), lo=-6.0)
    x_b, _, k_b, s_b, z, _ = block_scale_and_residual(x, 32)
    z_b = x_b - k_b[..., None] * jnp.log(2.0) - jnp.log(6.0)
    rhs_b = s_b[..., None] * 6.0 * jnp.exp(z_b)
    np.testing.assert_allclose(
        np.asarray(jnp.exp(x_b)), np.asarray(rhs_b), rtol=1e-10, atol=1e-12
    )
    s_full = jnp.repeat(s_b, 32, axis=-1)[..., : x.shape[-1]]
    np.testing.assert_allclose(
        np.asarray(jnp.exp(x)),
        np.asarray(s_full * 6.0 * jnp.exp(z)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_affine_thresholds_on_z_grid():
    tau, h = TAU_MMLU, H_MMLU
    z_vals = []
    expected = []
    for c in range(1, 7):
        lo = tau + (c - 1) / h
        hi = tau + c / h
        z_vals.append(0.5 * (lo + hi))
        expected.append(c)
    z_vals.append(tau - 2.0)
    expected.append(0)
    z_vals.append(tau + 7.0 / h + 2.0)
    expected.append(7)
    codes = affine_codes(jnp.asarray(z_vals), tau, h)
    np.testing.assert_array_equal(np.asarray(codes), np.asarray(expected, dtype=np.int32))

    z_grid = jnp.linspace(tau - 0.5, tau + 8.0 / h + 0.5, 400)
    codes_g = np.asarray(affine_codes(z_grid, tau, h))
    zg = np.asarray(z_grid)
    for c in range(1, 7):
        lo = tau + (c - 1) / h
        hi = tau + c / h
        interior = (zg >= lo) & (zg < hi)
        assert np.all(codes_g[interior] == c)


def test_clipping_extreme_z():
    tau, h = TAU_MEAN, H_MEAN
    z = jnp.asarray([-1e6, -50.0, 50.0, 1e6])
    codes = affine_codes(z, tau, h)
    np.testing.assert_array_equal(np.asarray(codes), np.array([0, 0, 7, 7], dtype=np.int32))


def test_shared_operand_numerator_denominator():
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    x = _random_shifted(k1, (6, 32))
    v = jax.random.normal(k2, (32, 4))
    op = efq_quantize(x, tau=TAU_BALANCE, h=H_BALANCE, block_size=32)
    num, den = apply_operand(op.ptilde, v)
    np.testing.assert_allclose(np.asarray(num), np.asarray(op.ptilde @ v), rtol=1e-12, atol=1e-12)
    ones = jnp.ones((32,), dtype=op.ptilde.dtype)
    np.testing.assert_allclose(
        np.asarray(den.squeeze(-1)), np.asarray(op.ptilde @ ones), rtol=1e-12, atol=1e-12
    )


def test_online_attention_vs_exact_and_uniform():
    key = jax.random.PRNGKey(4)
    nq, nk, d = 8, 16, 8
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (nq, d))
    k = jax.random.normal(kk, (nk, d))
    v = jax.random.normal(kv, (nk, d))
    exact = softmax_attention(q, k, v)
    efq_out, a, ell = online_attention(
        q, k, v, tile_k=8, tau=TAU_MMLU, h=H_MMLU, block_size=8
    )
    uniform = (jnp.ones((nq, nk)) / nk) @ v
    err_efq = jnp.linalg.norm(efq_out - exact) / jnp.linalg.norm(exact)
    err_uni = jnp.linalg.norm(uniform - exact) / jnp.linalg.norm(exact)
    assert np.isfinite(np.asarray(efq_out)).all()
    assert np.asarray(err_efq) < np.asarray(err_uni)
    assert np.asarray(err_efq) < 0.5
    # A/l is the normalized output; one-shot EFQ weights sum to 1.
    scale = jnp.sqrt(jnp.asarray(d, dtype=q.dtype))
    scores = (q @ k.T) / scale
    x = scores - jnp.max(scores, axis=-1, keepdims=True)
    p = efq_quantize(x, tau=TAU_MMLU, h=H_MMLU, block_size=8).ptilde
    w = p / jnp.sum(p, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(jnp.sum(w, axis=-1)), np.ones(nq), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(a / ell), np.asarray(efq_out), rtol=1e-12, atol=1e-12
    )


def test_causal_mask_maps_to_code_zero():
    key = jax.random.PRNGKey(5)
    nq = nk = 8
    d = 4
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (nq, d))
    k = jax.random.normal(kk, (nk, d))
    v = jax.random.normal(kv, (nk, d))
    causal = jnp.tril(jnp.ones((nq, nk), dtype=bool))
    scale = jnp.sqrt(jnp.asarray(d, dtype=q.dtype))
    scores = (q @ k.T) / scale
    scores = jnp.where(causal, scores, -jnp.inf)
    finite = jnp.where(jnp.isfinite(scores), scores, -1e30)
    x = scores - jnp.max(finite, axis=-1, keepdims=True)
    op = efq_quantize(x, tau=TAU_MMLU, h=H_MMLU, block_size=8)
    masked = np.asarray(~causal)
    assert np.all(np.asarray(op.codes)[masked] == 0)
    assert np.all(np.asarray(op.ptilde)[masked] == 0.0)
    out, _, _ = online_attention(q, k, v, tile_k=4, block_size=4, mask=causal)
    assert np.isfinite(np.asarray(out)).all()


def test_lut_variant():
    key = jax.random.PRNGKey(6)
    x = _random_shifted(key, (2, 32))
    op = efq_lut(x, block_size=32)
    j16 = np.asarray(op.j16)
    assert j16.min() >= 0 and j16.max() <= 15
    np.testing.assert_array_equal(np.asarray(op.codes), np.asarray(C16[op.j16]))
    q = np.asarray(C8[op.codes])
    assert set(np.unique(q)).issubset(C8_SET)
    tau_16 = float(np.log(1.0 / 24.0))
    h_16 = 14.0 / float(np.log(24.0))
    z = jnp.linspace(-8.0, 2.0, 200)
    j = np.asarray(lut_index(z))
    assert j.min() >= 0 and j.max() <= 15
    raw = np.floor((np.asarray(z) - tau_16) * h_16) + 1.0
    expect = np.clip(raw, 0, 15).astype(np.int32)
    np.testing.assert_array_equal(j, expect)


def test_historical_rescaling_and_flash_identity():
    key = jax.random.PRNGKey(7)
    nq, nk, d = 6, 16, 8
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (nq, d))
    k = jax.random.normal(kk, (nk, d))
    v = jax.random.normal(kv, (nk, d))
    exact_one = softmax_attention(q, k, v)
    exact_tiled, _, _ = online_attention(q, k, v, tile_k=8, exact=True)
    np.testing.assert_allclose(
        np.asarray(exact_tiled), np.asarray(exact_one), rtol=1e-12, atol=1e-12
    )

    scale = jnp.sqrt(jnp.asarray(d, dtype=q.dtype))
    s0 = (q @ k[:8].T) / scale
    s1 = (q @ k[8:].T) / scale
    m0 = jnp.max(s0, axis=-1, keepdims=True)
    m1 = jnp.maximum(m0, jnp.max(s1, axis=-1, keepdims=True))
    alpha = jnp.exp(m0 - m1)
    assert np.all(np.asarray(alpha) <= 1.0 + 1e-12)
    assert np.any(np.asarray(alpha) < 1.0 - 1e-8)

    efq_one, _, _ = online_attention(
        q, k, v, tile_k=nk, tau=TAU_MMLU, h=H_MMLU, block_size=8
    )
    efq_tile, _, _ = online_attention(
        q, k, v, tile_k=8, tau=TAU_MMLU, h=H_MMLU, block_size=8
    )
    rel = jnp.linalg.norm(efq_tile - efq_one) / jnp.linalg.norm(efq_one)
    assert np.asarray(rel) < 0.5
    uni = jnp.mean(v, axis=0, keepdims=True)
    err_one = jnp.linalg.norm(efq_one - exact_one) / jnp.linalg.norm(exact_one)
    err_tile = jnp.linalg.norm(efq_tile - exact_one) / jnp.linalg.norm(exact_one)
    err_uni = jnp.linalg.norm(uni - exact_one) / jnp.linalg.norm(exact_one)
    assert np.asarray(err_one) < np.asarray(err_uni)
    assert np.asarray(err_tile) < np.asarray(err_uni)
