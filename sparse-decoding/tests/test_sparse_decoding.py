"""Checks for decode-time calibration, OBS N:M pruning, and bitmask SpMV."""

import numpy as np
import jax.numpy as jnp

from sparse_decoding import (
    apply_obs_removal,
    bits_per_nonzero,
    bitmask_spmv,
    build_x_ar,
    calibration_discrepancy,
    gram_hessian,
    nm_prune,
    obs_saliency_and_update,
    pack_bitmask,
    pack_values,
    population_hessian,
    projected_reconstruction_loss,
    reconstruction_loss,
)


def test_prefill_columns_are_dropped():
    activations = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
    decode = jnp.array([False, False, True, True])
    gathered = build_x_ar(activations, decode)
    assert gathered.shape == (3, 2)
    assert jnp.allclose(gathered, activations[:, 2:])
    prompts = [activations, jnp.ones((3, 3), dtype=jnp.float64)]
    stacked = build_x_ar(prompts, [2, 1])
    assert stacked.shape[1] == 2 + 2
    assert jnp.allclose(stacked[:, :2], activations[:, 2:])
    assert jnp.allclose(stacked[:, 2:], jnp.ones((3, 2)))


def test_reconstruction_loss_sums_token_errors():
    weight = jnp.array([[1.0, -1.0], [0.5, 2.0]])
    sparse = jnp.array([[0.0, -1.0], [0.5, 0.0]])
    activations = jnp.array([[1.0, 0.0, 2.0], [0.5, 1.0, -1.0]])
    total = reconstruction_loss(weight, sparse, activations)
    manual = 0.0
    for col in range(activations.shape[1]):
        err = (weight - sparse) @ activations[:, col]
        manual += jnp.sum(err * err)
    assert jnp.allclose(total, manual)


def test_hessian_is_twice_gram():
    activations = jnp.array([[1.0, 0.5, -1.0], [0.0, 2.0, 0.25]])
    expected = 2.0 * activations @ activations.T
    assert jnp.allclose(gram_hessian(activations), expected)
    assert jnp.allclose(
        population_hessian(activations), expected / activations.shape[1]
    )


def test_obs_formula_and_loss_increase():
    row = jnp.array([0.4, -1.2, 0.7, 0.2], dtype=jnp.float64)
    activations = jnp.array(
        [
            [1.0, 0.2, 0.0, 0.1],
            [0.0, 1.0, 0.3, -0.2],
            [0.1, -0.4, 1.0, 0.2],
            [0.3, 0.1, -0.2, 1.0],
        ],
        dtype=jnp.float64,
    )
    hessian = gram_hessian(activations)
    h_inv = jnp.linalg.inv(hessian)
    index = 1
    saliency, delta = obs_saliency_and_update(row, h_inv, index)
    hand_l = row[index] ** 2 / (2.0 * h_inv[index, index])
    hand_delta = -(row[index] / h_inv[index, index]) * h_inv[:, index]
    assert jnp.allclose(saliency, hand_l)
    assert jnp.allclose(delta, hand_delta)
    updated, _, _ = apply_obs_removal(row, h_inv, index, freeze_prefix=False)
    loss = jnp.sum(((row - updated) @ activations) ** 2)
    assert jnp.allclose(loss, saliency, atol=1e-8)
    zeroed = row.at[index].set(0.0)
    zero_loss = jnp.sum(((row - zeroed) @ activations) ** 2)
    assert loss <= zero_loss + 1e-8


def test_nm_mask_keeps_largest_saliency_on_diagonal_hessian():
    # X = I => H = 2 I => L_q = w_q^2, so 2:4 keeps the two largest |w|.
    weight = jnp.array(
        [[1.0, 3.0, 2.0, 0.5], [0.2, -4.0, 0.1, 1.5]], dtype=jnp.float64
    )
    activations = jnp.eye(4, dtype=jnp.float64)
    pruned = nm_prune(weight, activations, n=2, m=4, damp=0.0)
    mask = np.asarray(pruned["mask"])
    assert mask[0].tolist() == [False, True, True, False]
    assert mask[1].tolist() == [False, True, False, True]
    assert np.all(mask.sum(axis=1) == 2)
    dense = weight * pruned["mask"]
    loss_obs = reconstruction_loss(weight, pruned["weight"], activations)
    loss_zero = reconstruction_loss(weight, dense, activations)
    assert loss_obs <= loss_zero + 1e-8


def test_obs_compensation_beats_zeroing_when_columns_correlate():
    weight = jnp.array([[1.5, -0.8, 0.4, 1.1, -0.3, 0.9, 0.2, -1.4]])
    rng = np.random.default_rng(0)
    activations = jnp.asarray(rng.normal(size=(8, 10)))
    pruned = nm_prune(weight, activations, n=2, m=4, damp=0.0)
    mask = pruned["mask"]
    assert jnp.all(mask.reshape(-1, 4).sum(axis=1) == 2)
    loss_obs = reconstruction_loss(weight, pruned["weight"], activations)
    loss_zero = reconstruction_loss(weight, weight * mask, activations)
    assert loss_obs < loss_zero - 1e-6


def test_bitmask_metadata_and_fixed_step_spmv():
    weight = jnp.arange(1, 65, dtype=jnp.float64).reshape(2, 32)
    activations = jnp.eye(32, dtype=jnp.float64) + 0.05
    pruned = nm_prune(weight, activations, n=2, m=4, damp=1e-6)
    mask = np.asarray(pruned["mask"])
    words = pack_bitmask(mask)
    assert words.shape == (2, 1)
    assert all(int(word).bit_count() == 16 for word in words.ravel())
    assert bits_per_nonzero(mask) == 2.0
    values = pack_values(pruned["weight"], mask)
    vector = jnp.linspace(-1.0, 1.0, 32, dtype=jnp.float64)
    sparse_y = bitmask_spmv(values, words, vector, steps=16)
    dense_y = pruned["weight"] @ vector
    assert jnp.allclose(sparse_y, dense_y, atol=1e-8)

    # 8:16 is also 50%, so each 32-bit word still has 16 set bits.
    pruned_816 = nm_prune(weight, activations, n=8, m=16, damp=1e-6)
    words_816 = pack_bitmask(np.asarray(pruned_816["mask"]))
    assert all(int(word).bit_count() == 16 for word in words_816.ravel())
    values_816 = pack_values(pruned_816["weight"], pruned_816["mask"])
    assert jnp.allclose(
        bitmask_spmv(values_816, words_816, vector, steps=16),
        pruned_816["weight"] @ vector,
        atol=1e-8,
    )


def test_bit_scan_matches_an_independent_column_walk():
    mask = np.zeros((1, 32), dtype=bool)
    mask[0, [1, 4, 5, 30]] = True
    weight = np.zeros((1, 32), dtype=np.float64)
    weight[0, 1] = 10.0
    weight[0, 4] = 3.0
    weight[0, 5] = 7.0
    weight[0, 30] = 2.0
    words = pack_bitmask(mask)
    values = pack_values(weight, mask)
    vector = np.arange(32, dtype=np.float64)
    got = np.asarray(bitmask_spmv(values, words, vector))
    expect = 10.0 * vector[1] + 3.0 * vector[4] + 7.0 * vector[5] + 2.0 * vector[30]
    assert np.allclose(got, expect)


def test_discrepancy_is_zero_on_a_match_and_scales():
    hessian = jnp.array(
        [[4.0, 0.5, 0.0], [0.5, 3.0, 0.2], [0.0, 0.2, 1.0]], dtype=jnp.float64
    )
    assert float(calibration_discrepancy(hessian, hessian, k=2)) < 1e-8
    # H_cal = 3 H_dec => normalized block is 3 I => operator norm of 3I - I is 2.
    assert jnp.allclose(calibration_discrepancy(hessian, 3.0 * hessian, k=2), 2.0)
    evals, evecs = jnp.linalg.eigh(hessian)
    top = evecs[:, jnp.argsort(evals)[::-1][:2]]
    delta = jnp.array([[0.2, -0.4, 0.1]])
    loss = projected_reconstruction_loss(delta, hessian, top)
    manual = 0.5 * jnp.trace(delta @ (top @ top.T) @ hessian @ (top @ top.T) @ delta.T)
    assert jnp.allclose(loss, manual)


def test_decode_hessian_differs_from_prefill_mixture():
    prefill = jnp.ones((4, 3))
    decode = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.2, -0.3]], dtype=jnp.float64
    )
    mixed = jnp.concatenate([prefill, decode], axis=1)
    mask = jnp.array([False, False, False, True, True])
    gathered = build_x_ar(mixed, mask)
    assert gathered.shape[1] == 2
    assert not jnp.allclose(gram_hessian(mixed), gram_hessian(gathered))


def test_prune_is_deterministic():
    weight = jnp.arange(16, dtype=jnp.float64).reshape(2, 8) / 7.0
    activations = jnp.arange(24, dtype=jnp.float64).reshape(8, 3) / 5.0 + 0.3
    first = nm_prune(weight, activations, n=2, m=4, damp=1e-4)
    second = nm_prune(weight, activations, n=2, m=4, damp=1e-4)
    assert jnp.allclose(first["weight"], second["weight"])
    assert jnp.array_equal(first["mask"], second["mask"])
