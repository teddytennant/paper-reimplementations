"""Latent solve: quadratic minimizer, ridge average, well-posedness."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rivat_fuse.solve import assemble_H, assemble_b, quadratic_objective, solve_latent
from rivat_fuse.trust import dense_P_from_side, init_trust_params


D = 6
EPS = 1e-4


def test_solve_matches_explicit_inverse():
    key = jax.random.PRNGKey(0)
    kH, kb = jax.random.split(key)
    A = jax.random.normal(kH, (D, D))
    H = A @ A.T + 0.1 * jnp.eye(D)
    b = jax.random.normal(kb, (D,))
    h = solve_latent(H, b)
    h_inv = jnp.linalg.inv(H) @ b
    assert jnp.allclose(h, h_inv, atol=1e-5, rtol=1e-5)


def test_solved_h_minimizes_quadratic():
    key = jax.random.PRNGKey(1)
    kH, kb, kpert = jax.random.split(key, 3)
    A = jax.random.normal(kH, (D, D))
    H = A @ A.T + jnp.eye(D)
    b = jax.random.normal(kb, (D,))
    h_star = solve_latent(H, b)
    obj_star = quadratic_objective(h_star, H, b)
    # Random perturbations should raise the strongly convex objective.
    noise = 0.05 * jax.random.normal(kpert, (8, D))
    objs = jax.vmap(lambda n: quadratic_objective(h_star + n, H, b))(noise)
    assert jnp.all(objs > obj_star)
    # Finite-difference gradient of 1/2 h^T H h - b^T h is H h - b ≈ 0 at h*.
    grad = H @ h_star - b
    assert jnp.allclose(grad, jnp.zeros_like(grad), atol=1e-5)


def test_ridge_average_identity_trust():
    """If T=0 and P_I = P_M = I, then h* = (z_I + z_M) / (2 + λ)."""
    key = jax.random.PRNGKey(2)
    z_I = jax.random.normal(key, (D,))
    z_M = jax.random.normal(jax.random.fold_in(key, 1), (D,))
    P_I = jnp.eye(D)
    P_M = jnp.eye(D)
    t_vec = jnp.zeros((D,))
    for lam in (0.0, 0.5, 2.0):
        H = assemble_H(P_I, P_M, lam)
        b = assemble_b(P_I, P_M, z_I, z_M, t_vec)
        h = solve_latent(H, b)
        expected = (z_I + z_M) / (2.0 + lam)
        assert jnp.allclose(h, expected, atol=1e-5, rtol=1e-5)
    # λ = 0 is the mean.
    H0 = assemble_H(P_I, P_M, 0.0)
    h0 = solve_latent(H0, assemble_b(P_I, P_M, z_I, z_M, t_vec))
    assert jnp.allclose(h0, 0.5 * (z_I + z_M), atol=1e-5)


def test_well_posedness_perturbation_bound():
    """||Δh|| is controlled by ||H^{-1}|| ||P_I|| ||δ|| (H well-conditioned)."""
    key = jax.random.PRNGKey(3)
    kt, kz, kd = jax.random.split(key, 3)
    trust = init_trust_params(kt, D, r_rank=3, s_dim=3)
    s = jnp.ones((3,)) * 0.4
    P_I = dense_P_from_side(trust["I"], s, eps=EPS)
    P_M = dense_P_from_side(trust["M"], s, eps=EPS)
    lam = 0.1
    H = assemble_H(P_I, P_M, lam)
    z_I = jax.random.normal(kz, (D,))
    z_M = jax.random.normal(jax.random.fold_in(kz, 1), (D,))
    t_vec = jnp.zeros((D,))
    h0 = solve_latent(H, assemble_b(P_I, P_M, z_I, z_M, t_vec))
    delta = 1e-3 * jax.random.normal(kd, (D,))
    h1 = solve_latent(H, assemble_b(P_I, P_M, z_I + delta, z_M, t_vec))
    dh = h1 - h0
    # Exact: Δh = H^{-1} P_I δ, so ||Δh|| <= ||H^{-1}||_2 ||P_I||_2 ||δ||.
    eigs_H = jnp.linalg.eigvalsh(H)
    cond = float(jnp.max(eigs_H) / jnp.min(eigs_H))
    assert cond < 1e6
    lip = float(jnp.linalg.norm(jnp.linalg.solve(H, P_I), ord=2))
    bound = lip * float(jnp.linalg.norm(delta))
    assert float(jnp.linalg.norm(dh)) <= bound + 1e-6
    assert float(jnp.linalg.norm(dh)) < 1.0
