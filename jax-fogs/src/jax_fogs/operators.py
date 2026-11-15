"""Grünwald–Letnikov operators, FOLTI simulation (eq. 3), Green matrices (eq. 4)."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammasgn


def gl_psi(alpha, j):
    """ψ(α, j) = Γ(j − α) / (Γ(−α) Γ(j + 1)).

    Broadcasts like NumPy. For α = 1, Γ(−1) is a pole so the binomial
    weights (1, −1, 0, 0, …) are used instead.
    """
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    j = jnp.asarray(j, dtype=jnp.float64)
    alpha, j = jnp.broadcast_arrays(alpha, j)
    near_one = jnp.abs(alpha - 1.0) < 1e-12
    alpha_safe = jnp.where(near_one, jnp.array(0.5, dtype=alpha.dtype), alpha)
    log_abs = gammaln(j - alpha_safe) - gammaln(-alpha_safe) - gammaln(j + 1.0)
    sgn = gammasgn(j - alpha_safe) * gammasgn(-alpha_safe) * gammasgn(j + 1.0)
    psi = sgn * jnp.exp(log_abs)
    psi = jnp.where(j == 0, jnp.ones_like(psi), psi)
    psi_one = jnp.where(j == 0, 1.0, jnp.where(j == 1, -1.0, 0.0))
    return jnp.where(near_one, psi_one, psi)


def _shifted_history(x, times, lags):
    """x[k − j] for k in `times`, j in `lags`, or 0 when j > k.

    x: (T, ...)  times: (S,)  lags: (L,)
    returns (S, L) + x.shape[1:]
    """
    mask = lags[None, :] <= times[:, None]
    idx = jnp.where(mask, times[:, None] - lags[None, :], 0)
    gathered = x[idx]
    shape = mask.shape + (1,) * (gathered.ndim - 2)
    return jnp.where(mask.reshape(shape), gathered, 0.0)


def fractional_difference(x, alpha):
    """Δ_α x_s for s = 1..T−1 (eq. 1), shape (T−1, n).

    x: (T, n) with rows x_0, …, x_{T−1}
    alpha: (n,) or scalar (broadcast to all coordinates)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    t_len, n = x.shape
    if alpha.ndim == 0:
        alpha = jnp.broadcast_to(alpha, (n,))
    j = jnp.arange(t_len)
    psi = gl_psi(alpha[None, :], j[:, None])
    times = jnp.arange(1, t_len)
    hist = _shifted_history(x, times, j)
    return jnp.sum(psi[None, :, :] * hist, axis=1)


def simulate_folti(A, alpha, x0, eta):
    """Recursion (eq. 3): x_{s+1} = A x_s − ∑_{j=1}^{s+1} Ψ(α, j) x_{s+1−j} + η_s.

    A: (n, n), alpha: (n,) or scalar, x0: (n,), eta: (t, n)
    returns x: (t+1, n)
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    t, n = eta.shape
    if alpha.ndim == 0:
        alpha = jnp.broadcast_to(alpha, (n,))
    j = jnp.arange(t + 1)
    psi = gl_psi(alpha[None, :], j[:, None])

    def step(x_hist, s):
        jj = jnp.arange(1, t + 1)
        mask = jj <= (s + 1)
        idx = jnp.where(mask, s + 1 - jj, 0)
        mem = jnp.sum(jnp.where(mask[:, None], psi[jj] * x_hist[idx], 0.0), axis=0)
        x_next = A @ x_hist[s] - mem + eta[s]
        x_hist = x_hist.at[s + 1].set(x_next)
        return x_hist, x_next

    x_hist = jnp.zeros((t + 1, n), dtype=jnp.float64).at[0].set(x0)
    x_hist, _ = jax.lax.scan(step, x_hist, jnp.arange(t))
    return x_hist


def green_matrices(A, alpha, horizon):
    """G_0, …, G_horizon (eq. 4), shape (horizon+1, n, n).

    G_0 = I, G_s = ∑_{j=0}^{s−1} A_j G_{s−1−j},
    A_0 = A + diag(α), A_j = −Ψ(α, j+1) for j ≥ 1.
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    n = A.shape[0]
    if alpha.ndim == 0:
        alpha = jnp.broadcast_to(alpha, (n,))
    horizon = int(horizon)
    j = jnp.arange(horizon)
    psi = gl_psi(alpha[None, :], (j + 1)[:, None])
    Aj = -jnp.eye(n, dtype=jnp.float64) * psi[:, None, :]
    Aj = Aj.at[0].set(A + jnp.diag(alpha))
    G = jnp.zeros((horizon + 1, n, n), dtype=jnp.float64)
    G = G.at[0].set(jnp.eye(n, dtype=jnp.float64))

    def step(G, s):
        s = s + 1
        jj = jnp.arange(horizon)
        mask = jj < s
        idx = jnp.where(mask, s - 1 - jj, 0)
        terms = jnp.einsum("jab,jbc->jac", Aj, G[idx])
        Gs = jnp.sum(jnp.where(mask[:, None, None], terms, 0.0), axis=0)
        G = G.at[s].set(Gs)
        return G, Gs

    G, _ = jax.lax.scan(step, G, jnp.arange(horizon))
    return G
