"""The sieve maximum likelihood estimator of eq (3.9).

Section 3.3 reduces the likelihood to the scalar energies, so

    L_n(xi) = -(1/n) sum_i log pi_xi(X_i) = -(1/n) sum_i log q_xi(V(X_i)),

and the estimator minimises that over the ReQU sieve X_m of eq (4.2).  We keep
the structural part of the sieve, kappa <= xi_theta <= K by eq (4.1), and drop
the sparsity constraint s_m and the Holder ball H^beta([0, U], H1); both are
there for the metric-entropy bounds of Section 5, not for the fit.

Each candidate needs a full solve of (3.5) plus the normalisation root find
(3.6), and optax differentiates straight through it.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from mkv_sieve.profile import (
    SolverConfig,
    profile_log_q,
    solve_profile,
)
from mkv_sieve.sieve import make_xi


def nll(params, energies, pot, kappa, K, tau=0.05, U=1.0, cfg=SolverConfig()):
    """L_n(xi_theta) of eq (3.9), evaluated at the observed energies V(X_i)."""
    xi = make_xi(params, kappa, K, tau, U)
    prof = solve_profile(xi, pot, kappa, K, cfg)
    return -jnp.mean(profile_log_q(prof, energies))


def fit_mle(
    params0,
    energies,
    pot,
    kappa: float,
    K: float,
    tau: float = 0.05,
    U: float = 1.0,
    cfg: SolverConfig = SolverConfig(),
    steps: int = 200,
    lr: float = 0.02,
):
    """Minimise (3.9) over the sieve with Adam.  Returns (params, loss trace)."""
    opt = optax.adam(lr)
    state = opt.init(params0)
    loss_and_grad = jax.value_and_grad(
        lambda p: nll(p, energies, pot, kappa, K, tau, U, cfg)
    )

    @jax.jit
    def step(carry, _):
        p, st = carry
        loss, g = loss_and_grad(p)
        updates, st = opt.update(g, st, p)
        p = optax.apply_updates(p, updates)
        return (p, st), loss

    (params, _), losses = jax.lax.scan(step, (params0, state), None, length=steps)
    return params, losses


def ise_on_interval(xi_hat: Callable, xi_true: Callable, r_lo, r_hi, n_grid: int = 501):
    """Mean squared error of the coefficient on I = [r_-, r_+], Section 8.

    The paper's ISE_{n,i,d,q}, a plain grid average over the interior interval
    that Remark 3.6 says the data identifies.
    """
    r = jnp.linspace(r_lo, r_hi, n_grid)
    return jnp.mean((xi_hat(r) - xi_true(r)) ** 2)


def l2_on_interval(xi_hat: Callable, xi_true: Callable, r_lo, r_hi, n_grid: int = 501):
    """||xi_hat - xi_0||_{L2(I)}, the square root of the integral over I."""
    return jnp.sqrt(ise_on_interval(xi_hat, xi_true, r_lo, r_hi, n_grid) * (r_hi - r_lo))


def kl_stationary(prof_true, prof_hat, pot):
    """KL(pi_0 || pi_hat), the quantity Section 4.2 and Section 5 bound.

    Written as a radial integral against the true profile's own quadrature grid,
    KL = int omega_d rho^{d-1} q_0(1 + rho^2) [log q_0 - log q_hat] drho.
    """
    s = prof_true.v_star + prof_true.rho**2
    y0 = profile_log_q(prof_true, s)
    y1 = profile_log_q(prof_hat, s)
    return jnp.sum(prof_true.w_rho * jnp.exp(y0) * (y0 - y1))
