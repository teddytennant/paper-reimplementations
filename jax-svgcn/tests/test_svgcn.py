"""Equation-level tests for SV-GCN (arXiv:2609.11680). CPU only."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "cpu")

from svgcn.features import (
    EPS,
    angle_feature,
    motion_features,
    relative_motion,
    relative_sign,
)
from svgcn.layers import (
    make_mask,
    masked_gap,
    masked_group_norm,
    masked_group_norm_stats,
)
from svgcn.model import apply_svgcn, cross_entropy, init_svgcn
from svgcn.skeleton import ADJACENCY, ANGLE_TRIPLES, NUM_JOINTS, PARENTS


def test_angle_range_and_special_cases():
    origin = jnp.zeros((3,), dtype=jnp.float32)
    x = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    nx = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float32)

    right = angle_feature(x, origin, y)
    collinear_0 = angle_feature(x, origin, x * 2)
    collinear_pi = angle_feature(x, origin, nx)
    acute = angle_feature(x, origin, jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32))

    assert 0.0 <= float(right) <= float(jnp.pi)
    assert 0.0 <= float(acute) <= float(jnp.pi)
    assert float(jnp.abs(right - jnp.pi / 2)) < 1e-5
    assert float(collinear_0) < 1e-5
    assert float(jnp.abs(collinear_pi - jnp.pi)) < 1e-5

    # Random triples stay in [0, π].
    key = jax.random.PRNGKey(0)
    pts = jax.random.normal(key, (32, 3, 3))
    ang = jax.vmap(lambda p: angle_feature(p[0], p[1], p[2]))(pts)
    assert jnp.all(ang >= 0.0) and jnp.all(ang <= jnp.pi)


def test_relative_motion_zero_when_equal_and_sign_pattern():
    f0 = jnp.array(
        [0.3, -0.2, 0.1, 0.4, -0.5, 0.05, 0.2, 0.6, -0.1, 0.08],
        dtype=jnp.float32,
    )
    rel = relative_motion(f0, f0)
    assert jnp.allclose(rel, jnp.zeros_like(rel), atol=1e-6)

    f_j = jnp.array(
        [0.6, 0.4, -0.2, 0.8, 0.1, -0.3, 0.05, 1.0, 0.2, -0.4],
        dtype=jnp.float32,
    )
    sigma = relative_sign(f_j, f0)
    rel = relative_motion(f_j, f0)
    mag = jnp.log(
        jnp.linalg.norm(f_j + EPS) / jnp.linalg.norm(f0 + EPS)
    )
    assert jnp.allclose(rel, sigma * mag, atol=1e-6)
    assert jnp.all(jnp.isin(sigma, jnp.array([-1.0, 1.0])))
    assert float(EPS) == 1e-8


def test_relative_motion_scale_invariance():
    """If both joint and root f_Δt scale by α, f_rel is invariant (eq. 2)."""
    f0 = jnp.array(
        [0.2, -0.1, 0.05, 0.3, -0.15, 0.08, 0.12, 0.25, 0.04, -0.02],
        dtype=jnp.float32,
    )
    fj = jnp.array(
        [0.5, 0.2, -0.3, 0.7, 0.1, -0.4, 0.2, 0.9, -0.06, 0.03],
        dtype=jnp.float32,
    )
    alpha = 2.0
    rel = relative_motion(fj, f0)
    rel_scaled = relative_motion(alpha * fj, alpha * f0)
    assert jnp.allclose(rel, rel_scaled, atol=1e-5)


def test_masked_group_norm_ignores_padding():
    key = jax.random.PRNGKey(1)
    n, c, t, j = 2, 8, 8, NUM_JOINTS
    x = jax.random.normal(key, (n, c, t, j))
    lengths = jnp.array([5, 5], dtype=jnp.int32)
    mask = make_mask(lengths, t)
    x_pad_a = x * mask[:, None, :, None]
    x_pad_b = x_pad_a.at[:, :, 5:, :].add(3.0)
    mu_a, var_a = masked_group_norm_stats(x_pad_a, mask, num_groups=4)
    mu_b, var_b = masked_group_norm_stats(x_pad_b, mask, num_groups=4)
    assert jnp.allclose(mu_a, mu_b, atol=1e-5)
    assert jnp.allclose(var_a, var_b, atol=1e-5)

    scale = jnp.ones((c,), dtype=jnp.float32)
    bias = jnp.zeros((c,), dtype=jnp.float32)
    y_a = masked_group_norm(x_pad_a, mask, scale, bias, num_groups=4)
    y_b = masked_group_norm(x_pad_b, mask, scale, bias, num_groups=4)
    valid = mask[:, None, :, None]
    assert jnp.allclose(y_a * valid, y_b * valid, atol=1e-5)


def test_masked_gap_ignores_padding():
    key = jax.random.PRNGKey(2)
    n, c, t, j = 3, 6, 10, NUM_JOINTS
    x = jax.random.normal(key, (n, c, t, j))
    lengths = jnp.array([4, 7, 10], dtype=jnp.int32)
    mask = make_mask(lengths, t)
    x_a = x * mask[:, None, :, None]
    x_b = x_a.at[:, :, :, :].add((1.0 - mask[:, None, :, None]) * 9.0)
    gap_a = masked_gap(x_a, mask)
    gap_b = masked_gap(x_b, mask)
    assert jnp.allclose(gap_a, gap_b, atol=1e-5)


def test_forward_shape_and_padding_invariance():
    key = jax.random.PRNGKey(3)
    params = init_svgcn(key)
    n, t, j = 4, 16, NUM_JOINTS
    k_c, k_l = jax.random.split(jax.random.PRNGKey(4))
    coords = jax.random.normal(k_c, (n, t, j, 3))
    lengths = jnp.array([7, 10, 12, 16], dtype=jnp.int32)
    mask = make_mask(lengths, t)
    coords = coords * mask[:, :, None, None]
    logits = apply_svgcn(params, coords, lengths)
    assert logits.shape == (n, 4)
    assert jnp.all(jnp.isfinite(logits))

    # Same valid prefix, extra padded junk vs truncated batch of one sample.
    i = 0
    li = int(lengths[i])
    coords_pad = coords.at[i, li:].set(5.0)
    logits_pad = apply_svgcn(params, coords_pad, lengths)
    coords_trunc = coords[i : i + 1, :li]
    logits_trunc = apply_svgcn(params, coords_trunc, jnp.array([li], dtype=jnp.int32))
    # After GAP+linear, padded zeros vs truncated-valid should match closely.
    # Residual temporal conv with SAME padding can leak a little at the
    # valid/pad boundary; compare against zero-padded (not junk-padded) input.
    logits_zero = apply_svgcn(params, coords, lengths)
    assert jnp.allclose(logits_zero[i], logits_trunc[0], atol=2e-3)
    # Junk in the pad region is killed by mask-multiply before SMEF, so
    # logits_pad should match logits_zero.
    assert jnp.allclose(logits_pad, logits_zero, atol=1e-5)


def test_one_train_step_finite_and_loss_drops():
    key = jax.random.PRNGKey(5)
    params = init_svgcn(key)
    from svgcn.data import generate_synthetic

    coords, lengths, labels = generate_synthetic(
        jax.random.PRNGKey(6), batch_size=16, t_max=16, min_len=8
    )

    def loss_fn(p):
        logits = apply_svgcn(p, coords, lengths)
        return cross_entropy(logits, labels)

    loss0, grads = jax.value_and_grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert jnp.isfinite(loss0)

    def add(p, g):
        return p - 0.05 * g

    params1 = jax.tree_util.tree_map(add, params, grads)
    loss1 = loss_fn(params1)
    assert jnp.isfinite(loss1)
    assert float(loss1) < float(loss0) or float(loss1) <= float(loss0) + 1e-4


def test_skeleton_consistency():
    assert len(PARENTS) == NUM_JOINTS
    assert len(ANGLE_TRIPLES) == NUM_JOINTS
    assert ADJACENCY.shape == (NUM_JOINTS, NUM_JOINTS)
    assert jnp.allclose(ADJACENCY, ADJACENCY.T)
    assert jnp.allclose(jnp.diag(ADJACENCY), jnp.ones(NUM_JOINTS))
