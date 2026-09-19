import jax.numpy as jnp

from jax_fogs import (
    alpha_grid,
    fogs,
    fogs_commensurate,
    fractional_difference,
    simulate_folti,
)


def _plant(t=80):
    A = jnp.array([[0.25, 0.10], [-0.05, 0.20]])
    alpha = jnp.array([0.2, 0.4])
    x0 = jnp.array([1.0, -0.7])
    traj = simulate_folti(A, alpha, x0, jnp.zeros((t, 2)))
    grid = alpha_grid(0.05, 0.55, 11)
    return A, alpha, traj, grid


def test_noise_free_identification():
    A, alpha, traj, grid = _plant()
    assert jnp.max(jnp.abs(jnp.linalg.eigvals(A))) < 0.5
    assert jnp.any(jnp.isclose(grid, 0.2))
    assert jnp.any(jnp.isclose(grid, 0.4))
    a_hat, A_hat = fogs(traj, grid, ridge=1e-6)
    assert jnp.allclose(a_hat, alpha, atol=1e-12)
    assert jnp.allclose(A_hat, A, atol=1e-4)


def test_ols_matches_lstsq_at_true_alpha():
    A, alpha, traj, _ = _plant()
    X = traj[:-1].T
    dX = fractional_difference(traj, alpha).T
    a_closed = dX @ X.T @ jnp.linalg.inv(X @ X.T)
    a_lstsq = jnp.linalg.lstsq(X.T, dX.T, rcond=None)[0].T
    assert jnp.allclose(a_closed, a_lstsq, atol=1e-8)
    assert jnp.allclose(a_closed, A, atol=1e-8)


def test_loss_splits_into_row_losses():
    A, alpha, traj, _ = _plant()
    dX = fractional_difference(traj, alpha)
    resid = dX - traj[:-1] @ A.T
    total = jnp.sum(resid**2)
    row = jnp.sum(resid**2, axis=0)
    assert jnp.allclose(total, jnp.sum(row), atol=1e-14)


def test_commensurate_hits_shared_alpha():
    A = jnp.array([[0.25, 0.10], [-0.05, 0.20]])
    alpha = 0.3
    x0 = jnp.array([1.0, -0.7])
    traj = simulate_folti(A, alpha, x0, jnp.zeros((80, 2)))
    grid = alpha_grid(0.05, 0.55, 11)
    assert jnp.any(jnp.isclose(grid, 0.3))
    a_hat, A_hat = fogs_commensurate(traj, grid, ridge=1e-6)
    assert jnp.allclose(a_hat, alpha, atol=1e-12)
    assert jnp.allclose(A_hat, A, atol=1e-4)


def test_ridge_uses_xxT_plus_lambda_I():
    _, alpha, traj, _ = _plant()
    X = traj[:-1].T
    dX = fractional_difference(traj, alpha).T
    lam = 1.0
    gram = X @ X.T + lam * jnp.eye(2)
    A_ridge = dX @ X.T @ jnp.linalg.inv(gram)
    a_hat, A_hat = fogs(traj, alpha[:, None], ridge=lam)
    assert jnp.allclose(a_hat, alpha, atol=1e-12)
    assert jnp.allclose(A_hat, A_ridge, atol=1e-8)
    A_ols = dX @ X.T @ jnp.linalg.inv(X @ X.T)
    assert jnp.max(jnp.abs(A_hat - A_ols)) > 1e-4
