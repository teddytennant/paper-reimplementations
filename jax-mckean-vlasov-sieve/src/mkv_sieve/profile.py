"""The forward map xi -> pi_xi of Proposition 3.4, eq (3.4) to (3.6).

The profile q_xi solves the scalar autonomous ODE (3.5)

    q'(s) = -q(s) xi(q(s)),      s > v_star,

and its initial value a_xi = q_xi(v_star) is fixed by the normalisation (3.6),
int_{v_star}^inf q_xi(s) m_V(s) ds = 1.

Because (3.5) is autonomous, every solution is the same trajectory read off at a
different starting point, so there is no need to re-integrate the ODE once per
candidate a.  We integrate the trajectory once, from the top of the bracket that
Remark 3.6 puts on a_xi, and the family is then indexed by a single shift t in
the energy variable.  Concretely we work with y = log q and store

    e(y) = int_y^{y_top} dv / xi(e^v),

which is the primitive g_xi of eq (3.3) measured downward from exp(y_top): under
u = log r, g_xi(r) = int_0^{log r} dv / xi(e^v).  Then eq (3.4), q_xi(s) =
g_xi^{-1}(mu_xi - s), reads

    log q_xi(s) = e^{-1}(s - v_star + t),      t = v_star - mu_xi + g_xi(e^{y_top}),

and the normalisation (3.6) becomes a monotone scalar equation in t.  That gets
one cumulative quadrature plus a bisection over interpolations, no per-iteration
ODE solve, and it makes the whole map cheap enough to differentiate through in
an optax loop.

The root find is a bisection wrapped in stop_gradient followed by one Newton
step taken with live gradients, so the returned t is the bisection's value
numerically but carries the implicit-function derivative dt/dtheta =
-(dN/dtheta)/(dN/dt).
"""

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from mkv_sieve.quad import cum_quad4, hermite_interp, simpson_weights


class SolverConfig(NamedTuple):
    """Grid sizes for the forward map.

    n_y      nodes on the log-density grid the trajectory is tabulated on
    n_rho    Simpson nodes in the radial variable, must be odd
    n_bisect bisection steps for the normalisation (3.6)
    tail     radial cutoff, R = sqrt(tail / kappa); the envelope (3.7) puts the
             discarded mass below exp(-tail)
    head     safety factor widening the a-bracket of Remark 3.6 on both sides
    """

    n_y: int = 6001
    n_rho: int = 1025
    n_bisect: int = 60
    tail: float = 38.0
    head: float = 1.5


class Profile(NamedTuple):
    """A tabulated solution of (3.5) together with the shift fixed by (3.6)."""

    e: jnp.ndarray  # increasing, e[0] = 0
    y: jnp.ndarray  # log q at the nodes, decreasing
    dy: jnp.ndarray  # dy/de = -xi(exp(y))
    t: jnp.ndarray  # scalar shift; log q(s) = y(s - v_star + t)
    rho: jnp.ndarray  # uniform radial grid on [0, R]
    w_rho: jnp.ndarray  # Simpson weights times the radial volume factor
    cdf: jnp.ndarray  # cdf of the radius under pi_xi, on the rho grid
    pdf_rho: jnp.ndarray  # its density, omega_d rho^{d-1} q_xi(1 + rho^2)
    v_star: jnp.ndarray


def solve_profile(
    xi: Callable[[jnp.ndarray], jnp.ndarray],
    pot,
    kappa: float,
    K: float,
    cfg: SolverConfig = SolverConfig(),
) -> Profile:
    """Solve (3.5) and pick a_xi by the normalisation (3.6).

    xi must satisfy kappa <= xi <= K on [0, inf), Assumption 3.3, and must
    accept an array.  The bracket for the bisection is Remark 3.6's
    Z_kappa^{-1} <= a_xi <= Z_K^{-1}, widened by cfg.head.
    """
    v_star = pot.v_star
    R = math.sqrt(cfg.tail / kappa)

    # log-a bracket from Remark 3.6, widened so the shift is strictly interior
    log_a_hi = -pot.log_Z(K) + math.log(cfg.head)
    log_a_lo = -pot.log_Z(kappa) - math.log(cfg.head)
    y_top = log_a_hi
    t_max = (log_a_hi - log_a_lo) / kappa * 1.05

    # the trajectory has to reach energy R^2 past the smallest admissible a
    e_max = R * R + t_max
    y_span = K * e_max * 1.02  # de/d|y| >= 1/K, so this covers e_max
    hy = y_span / (cfg.n_y - 1)
    yg = y_top - hy * jnp.arange(cfg.n_y)

    xi_vals = xi(jnp.exp(yg))
    e = cum_quad4(1.0 / xi_vals, hy)
    dy = -xi_vals

    h_rho = R / (cfg.n_rho - 1)
    rho = h_rho * jnp.arange(cfg.n_rho)
    vol = pot.radial_weight(rho)
    w_rho = simpson_weights(cfg.n_rho, h_rho) * vol
    u_rho = rho * rho

    def log_mass(t, e_, y_, dy_):
        """log of int q_xi(V(x)) dx with the shift t, eq (3.6)."""
        yy = hermite_interp(e_, y_, dy_, u_rho + t)
        return jnp.log(jnp.sum(w_rho * jnp.exp(yy)))

    y = yg
    eb, yb, dyb = jax.lax.stop_gradient((e, y, dy))

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        pos = log_mass(mid, eb, yb, dyb) > 0.0
        return (jnp.where(pos, mid, lo), jnp.where(pos, hi, mid))

    lo, hi = jax.lax.fori_loop(
        0, cfg.n_bisect, body, (jnp.asarray(0.0), jnp.asarray(t_max))
    )
    t0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    slope = jax.lax.stop_gradient(
        jax.grad(lambda tt: log_mass(tt, eb, yb, dyb))(t0)
    )
    t = t0 - log_mass(t0, e, y, dy) / slope

    pdf_rho = vol * jnp.exp(hermite_interp(e, y, dy, u_rho + t))
    cdf = cum_quad4(pdf_rho, h_rho)
    total = cdf[-1]
    cdf = cdf / total
    pdf_rho = pdf_rho / total

    return Profile(
        e=e,
        y=y,
        dy=dy,
        t=t,
        rho=rho,
        w_rho=w_rho,
        cdf=cdf,
        pdf_rho=pdf_rho,
        v_star=jnp.asarray(v_star),
    )


def profile_log_q(prof: Profile, s):
    """log q_xi(s) for energies s >= v_star, the profile of eq (3.5)."""
    return hermite_interp(prof.e, prof.y, prof.dy, s - prof.v_star + prof.t)


def profile_q(prof: Profile, s):
    return jnp.exp(profile_log_q(prof, s))


def profile_a(prof: Profile):
    """a_xi = q_xi(v_star), the mode of the stationary density."""
    return jnp.exp(profile_log_q(prof, prof.v_star))


def profile_log_pi(prof: Profile, pot, x):
    """log pi_xi(x) = log q_xi(V(x)), eq (3.4) with Proposition 3.4."""
    return profile_log_q(prof, pot.V(x))


def profile_pi(prof: Profile, pot, x):
    return jnp.exp(profile_log_pi(prof, pot, x))


def profile_energy_cdf(prof: Profile, s):
    """P(V(X) <= s) under pi_xi, from f_xi(s) = q_xi(s) m_V(s) of Section 3.3."""
    r = jnp.sqrt(jnp.maximum(s - prof.v_star, 0.0))
    return jnp.clip(hermite_interp(prof.rho, prof.cdf, prof.pdf_rho, r), 0.0, 1.0)


def profile_mass(prof: Profile):
    """int pi_xi over R^d on the solver's own radial grid.  Should be 1, eq (3.6)."""
    yy = hermite_interp(prof.e, prof.y, prof.dy, prof.rho**2 + prof.t)
    return jnp.sum(prof.w_rho * jnp.exp(yy))
