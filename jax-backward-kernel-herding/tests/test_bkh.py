"""Identities and agreement tests for Backward Kernel Herding."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from bkh.bkh import (
    backward_kernel_herding,
    g_remaining_mean,
    j_removal,
    kernelized_backward_herding,
)
from bkh.kernels import linear_kernel_matrix, mmd2_features


def _blob(n=24, d=3, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n, d), dtype=jnp.float64)


def test_j_identity_for_every_r():
    phi = _blob(n=20, d=4, seed=1)
    mu = jnp.mean(phi, axis=0)
    n = phi.shape[0]
    for r in range(n):
        j_val, scaled = j_removal(phi, r, mu=mu)
        assert jnp.allclose(j_val, scaled, rtol=0.0, atol=1e-12)
        g = g_remaining_mean(phi, r, mu=mu)
        diff = mu - g
        j_direct = jnp.sum(diff * diff)
        assert jnp.allclose(j_direct, j_val, rtol=0.0, atol=1e-12)


def test_first_removed_is_closest_to_mean():
    phi = _blob(n=18, d=2, seed=2)
    mu = jnp.mean(phi, axis=0)
    diff = phi - mu[None, :]
    sq = jnp.sum(diff * diff, axis=1)
    expected = int(jnp.argmin(sq))
    _remaining, removed, _eps, _mu = backward_kernel_herding(phi, target_size=n_keep(phi, 5))
    assert int(removed[0]) == expected


def n_keep(phi, n_remove):
    return int(phi.shape[0] - n_remove)


def test_mmd_identity_after_m_removals():
    phi = _blob(n=16, d=3, seed=3)
    n = phi.shape[0]
    m = 6
    remaining, removed, eps, mu = backward_kernel_herding(phi, target_size=n - m)
    # Residual definition: eps_m = sum_{R_m} Phi - m mu_n (eq 15 / 3.2.1).
    eps_def = jnp.sum(phi[removed], axis=0) - m * mu
    assert jnp.allclose(eps, eps_def, rtol=0.0, atol=1e-10)
    mu_m = jnp.mean(phi[remaining], axis=0)
    mmd_left = jnp.sum((mu_m - mu) * (mu_m - mu))
    mmd_right = jnp.sum(eps * eps) / ((n - m) ** 2)
    assert jnp.allclose(mmd_left, mmd_right, rtol=0.0, atol=1e-12)
    assert jnp.allclose(mmd_left, mmd2_features(phi[remaining], phi), rtol=0.0, atol=1e-12)


def test_conservation_identity():
    phi = _blob(n=15, d=4, seed=4)
    n = phi.shape[0]
    remaining, removed, _eps, mu = backward_kernel_herding(phi, target_size=9)
    total = jnp.sum(phi[remaining], axis=0) + jnp.sum(phi[removed], axis=0)
    assert jnp.allclose(n * mu, total, rtol=0.0, atol=1e-10)
    assert jnp.allclose(n * mu, jnp.sum(phi, axis=0), rtol=0.0, atol=1e-10)


def test_kernelized_matches_explicit_linear():
    phi = _blob(n=14, d=3, seed=5)
    # Jitter so squared distances to the mean are not tied.
    phi = phi + jnp.linspace(0.0, 0.02, phi.shape[0])[:, None]
    n = phi.shape[0]
    target = 6
    rem_e, rem_order_e, _eps, _mu = backward_kernel_herding(phi, target)
    k = linear_kernel_matrix(phi)
    rem_k, rem_order_k = kernelized_backward_herding(k, target)
    assert jnp.array_equal(rem_order_e, rem_order_k)
    assert set(int(i) for i in rem_e) == set(int(i) for i in rem_k)
    assert set(int(i) for i in rem_e) == set(range(n)) - set(int(i) for i in rem_order_e)


def test_interior_removal_keeps_mmd_smaller_than_outliers_first():
    # Interior points go first (Section 3.2.1). After a modest number of
    # removals, remaining MMD is smaller than after dropping farthest-from-mean
    # points. Greedy BKH can later cluster removals, so m is kept small.
    key = jax.random.PRNGKey(9)
    x = jax.random.normal(key, (40, 2), dtype=jnp.float64)
    n = x.shape[0]
    m = 4
    remaining, _removed, _eps, mu = backward_kernel_herding(x, target_size=n - m)
    mmd_bkh = mmd2_features(x[remaining], x)

    diff = x - mu[None, :]
    sq = jnp.sum(diff * diff, axis=1)
    farthest = jnp.argsort(-sq)[:m]
    keep_mask = jnp.ones((n,), dtype=bool).at[farthest].set(False)
    mmd_far = mmd2_features(x[keep_mask], x)
    assert float(mmd_bkh) < float(mmd_far)
