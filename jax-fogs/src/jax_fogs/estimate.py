"""FO-GS (Algorithm 1) and a commensurate shared-α search."""

import jax
import jax.numpy as jnp

from jax_fogs.operators import fractional_difference, gl_psi


def alpha_grid(low, high, num):
    """M equally spaced points on [low, high], including both endpoints.

    Paper Algorithm 1 uses α̲ + k ε for k = 1..M with ε = (ᾱ − α̲)/(M+1),
    which skips the endpoints. Experiments used equally spaced grids such
    as 20 points on [0.05, 0.55]; this helper matches that closed linspace.
    """
    return jnp.linspace(low, high, int(num), dtype=jnp.float64)


def _gram(X, ridge):
    n = X.shape[0]
    return X @ X.T + ridge * jnp.eye(n, dtype=X.dtype)


def _row_ols(delta, X, gram):
    """â = Δ Xᵀ (X Xᵀ + λ I)^{−1} (eq. 7 with ridge).

    delta: (M, t) or (t,)   X: (n, t)   returns (M, n) or (n,)
    """
    squeezed = delta.ndim == 1
    if squeezed:
        delta = delta[None, :]
    a_hat = jax.scipy.linalg.solve(gram, X @ delta.T, assume_a="pos").T
    if squeezed:
        return a_hat[0]
    return a_hat


def _profile_row(x_i, X, gram, grid):
    """Grid search for one coordinate (eqs. 5, 7, 8)."""
    t_len = x_i.shape[0]
    j = jnp.arange(t_len)
    psi = gl_psi(grid[:, None], j[None, :])
    times = jnp.arange(1, t_len)
    mask = j[None, :] <= times[:, None]
    idx = jnp.where(mask, times[:, None] - j[None, :], 0)
    x_hist = jnp.where(mask, x_i[idx], 0.0)
    delta = psi @ x_hist.T
    a_hat = _row_ols(delta, X, gram)
    loss = jnp.sum((delta - a_hat @ X) ** 2, axis=1)
    best = jnp.argmin(loss)
    return grid[best], a_hat[best]


def fogs(traj, grid, ridge=1e-6):
    """Heterogeneous FO-GS (Algorithm 1).

    traj: (t+1, n) with rows x_0, …, x_t
    grid: (M,) shared search grid, or (n, M) per-coordinate grids
    ridge: λ on X Xᵀ; paper synthetic default 1e-6
    returns α̂ (n,), Â (n, n) with rows â_i
    """
    traj = jnp.asarray(traj, dtype=jnp.float64)
    grid = jnp.asarray(grid, dtype=jnp.float64)
    ridge = jnp.asarray(ridge, dtype=jnp.float64)
    n = traj.shape[1]
    X = traj[:-1].T
    gram = _gram(X, ridge)
    if grid.ndim == 1:
        grids = jnp.broadcast_to(grid, (n, grid.shape[0]))
    else:
        grids = grid

    def one_row(x_i, g):
        return _profile_row(x_i, X, gram, g)

    alpha_hat, A_hat = jax.vmap(one_row)(traj.T, grids)
    return alpha_hat, A_hat


def fogs_commensurate(traj, grid, ridge=1e-6):
    """Shared-α search, then full OLS Â(α) = (Δ_α X) Xᵀ (X Xᵀ + λ I)^{−1}."""
    traj = jnp.asarray(traj, dtype=jnp.float64)
    grid = jnp.asarray(grid, dtype=jnp.float64)
    ridge = jnp.asarray(ridge, dtype=jnp.float64)
    X = traj[:-1].T
    gram = _gram(X, ridge)

    def one_alpha(alpha):
        dX = fractional_difference(traj, alpha).T
        A = _row_ols(dX, X, gram)
        loss = jnp.sum((dX - A @ X) ** 2)
        return loss, A

    losses, As = jax.vmap(one_alpha)(grid)
    best = jnp.argmin(losses)
    return grid[best], As[best]
