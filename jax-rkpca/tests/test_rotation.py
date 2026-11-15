"""Rotation construction, Type I / Type II, and optimized apply."""

from __future__ import annotations

import numpy as np

from jax_rkpca.rotation import (
    apply_plane_rotation,
    rotation_matrix,
    type_i_vectors,
    type_ii_index,
    type_ii_vector,
)
from jax_rkpca.update import init_state, orthonormalize, step


def _rand_orthonormal(M, d, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(M, d)))
    return Q


def test_A_is_rotation_not_reflection():
    rng = np.random.default_rng(4)
    M = 6
    v = rng.normal(size=(M,))
    w = rng.normal(size=(M,))
    A = np.asarray(rotation_matrix(v, w, alpha=0.4))
    np.testing.assert_allclose(A.T @ A, np.eye(M), atol=1e-8)
    det = float(np.linalg.det(A))
    assert det > 0
    np.testing.assert_allclose(det, 1.0, atol=1e-8)


def test_full_rotation_lands_v_parallel_to_w():
    rng = np.random.default_rng(5)
    M = 7
    v = rng.normal(size=(M,))
    w = rng.normal(size=(M,))
    A = np.asarray(rotation_matrix(v, w, alpha=1.0))
    Av = A @ v
    cos = float(np.dot(Av, w) / (np.linalg.norm(Av) * np.linalg.norm(w)))
    assert cos > 1.0 - 1e-8


def test_type_i_alpha_zero_leaves_gamma_unchanged():
    rng = np.random.default_rng(6)
    M, d = 8, 3
    Gamma = _rand_orthonormal(M, d, seed=6)
    h = rng.normal(size=(M,))
    mu = rng.normal(size=(M,))
    v, w = type_i_vectors(h, mu, Gamma)
    G_new = np.asarray(apply_plane_rotation(Gamma, v, w, alpha=0.0))
    np.testing.assert_allclose(G_new, Gamma, atol=1e-12)


def test_type_ii_picks_max_cosine_column():
    rng = np.random.default_rng(7)
    M, d = 9, 4
    Gamma = _rand_orthonormal(M, d, seed=7)
    # Make w closest (signed cosine) to column 2.
    w = 4.0 * Gamma[:, 2] + 0.3 * Gamma[:, 0] + 0.1 * rng.normal(size=(M,))
    i_star = int(type_ii_index(w, Gamma))
    dots = Gamma.T @ w
    norms = np.linalg.norm(Gamma, axis=0)
    expected = int(np.argmax(dots / norms))
    assert i_star == expected
    v = np.asarray(type_ii_vector(w, Gamma))
    np.testing.assert_allclose(v, Gamma[:, expected])


def test_optimized_apply_matches_dense_A():
    rng = np.random.default_rng(8)
    M, d = 10, 3
    Gamma = _rand_orthonormal(M, d, seed=8)
    v = rng.normal(size=(M,))
    w = rng.normal(size=(M,))
    alpha = 0.37
    A = np.asarray(rotation_matrix(v, w, alpha))
    G_dense = A @ Gamma
    G_fast = np.asarray(apply_plane_rotation(Gamma, v, w, alpha))
    np.testing.assert_allclose(G_fast, G_dense, rtol=1e-10, atol=1e-10)


def test_gamma_orthonormal_after_step():
    rng = np.random.default_rng(9)
    M, d = 8, 3
    state = init_state(M, d)
    h = rng.normal(size=(M,))
    new_state, _, _ = step(state, h, eta=0.1, alpha0=0.5)
    G = np.asarray(new_state.Gamma)
    np.testing.assert_allclose(G.T @ G, np.eye(d), atol=1e-5)


def test_rotation_preserves_orthonormality_without_reorth():
    rng = np.random.default_rng(10)
    M, d = 8, 3
    Gamma = _rand_orthonormal(M, d, seed=10)
    v = rng.normal(size=(M,))
    w = rng.normal(size=(M,))
    G_rot = np.asarray(apply_plane_rotation(Gamma, v, w, alpha=0.6))
    np.testing.assert_allclose(G_rot.T @ G_rot, np.eye(d), atol=1e-8)
    # sanity: orthonormalize is a no-op up to sign
    G_q = np.asarray(orthonormalize(G_rot))
    np.testing.assert_allclose(np.abs(G_q.T @ G_rot), np.eye(d), atol=1e-8)
