"""The lifted simplex, the square-root entropy regularizer, and the map Q_m.

Section 3.1 lifts player i's simplex to

    lifted_simplex(m) = {(lam, y) : 0 <= lam <= 1, y in lam * simplex(m)},

where (lam, y) plays x = y / lam. Section 3.2 puts

    Gamma_m = (log m)^2 + 2 log m + 2,      Xi_m = 10^4 (1 + Gamma_m),
    psi_m(x) = sum_a x(a) log x(a),
    psitilde_m(lam, y) = -sqrt(1 - lam) + sqrt(lam) [psi_m(y/lam) - 2 Gamma_m - 3],

with psitilde_m(0, 0) = -1, and defines

    Q_m(theta) = argmax over the lifted simplex of <theta, y> - psitilde_m(lam, y).

Lemma 3.1 solves that maximization. Fixing lam and maximizing over x gives
x = softmax(sqrt(lam) theta), so the inner problem is Hedge at learning rate
sqrt(lam); substituting back and differentiating in lam leaves the scalar
stationarity condition

    1/sqrt(1-lam) - (2 Gamma_m + 3 + logsumexp(sqrt(lam) theta))/sqrt(lam)
        - <theta, softmax(sqrt(lam) theta)> = 0,

which Lemma 3.1 proves is strictly increasing on (0,1) with limits -inf and
+inf at the endpoints. Remark 3.2 says to localize the root by binary search,
which is what `lifted_argmax` does.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def gamma_const(m: int) -> float:
    """Gamma_m of Section 3.2, a bound on Var_{a~x}[-log x(a)] over the simplex."""
    lm = math.log(m)
    return lm**2 + 2.0 * lm + 2.0


def xi_const(m: int) -> float:
    """Xi_m = 10^4 (1 + Gamma_m), Section 3.2."""
    return 1e4 * (1.0 + gamma_const(m))


def theoretical_learning_rate(num_players: int, num_actions: int) -> float:
    """eta = 2^-94 / (N^20 Xi_max) from Theorem 3.3.

    Around 1e-70 for a two-player two-action game. It is a proof constant, not
    a number you can run, so `dynamics.run_game` defaults to something usable
    and this stays here for reference.
    """
    return 2.0**-94 / (num_players**20 * xi_const(num_actions))


def entropy(x: jnp.ndarray) -> jnp.ndarray:
    """psi_m(x) = sum_a x(a) log x(a), with 0 log 0 = 0."""
    return jnp.sum(jnp.where(x > 0, x * jnp.log(jnp.where(x > 0, x, 1.0)), 0.0), axis=-1)


def psi_tilde(lam: jnp.ndarray, y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """psitilde_m of Section 3.2, evaluated at (lam, y) with y = lam x."""
    lam = jnp.asarray(lam)
    x = y / jnp.where(lam > 0, lam, 1.0)[..., None]
    body = jnp.sqrt(lam) * (entropy(x) - 2.0 * gamma - 3.0)
    return jnp.where(lam > 0, -jnp.sqrt(jnp.maximum(1.0 - lam, 0.0)) + body, -1.0)


def stationarity(lam: jnp.ndarray, theta: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """The strictly increasing function of Lemma 3.1 whose zero is lambda_m(theta)."""
    s = jnp.sqrt(lam)
    z = s[..., None] * theta
    lse = logsumexp(z, axis=-1)
    x = jax.nn.softmax(z, axis=-1)
    return 1.0 / jnp.sqrt(1.0 - lam) - (2.0 * gamma + 3.0 + lse) / s - jnp.sum(theta * x, axis=-1)


def lifted_argmax(theta: jnp.ndarray, gamma: float, n_bisect: int = 60):
    """Q_m(theta) = (lam, lam x), by bisection on the Lemma 3.1 stationarity function.

    Returns (lam, x) rather than (lam, y); y = lam * x.

    60 bisection steps shrink the bracket by 2^-60, well under float64 eps
    relative to lam. The bracket endpoints are pulled off 0 and 1 by `pad` so
    that 1/sqrt(lam) and 1/sqrt(1-lam) stay finite; the stationarity function
    is around -1e7 at the low end and +1e7 at the high end for any theta this
    code produces, so the root is inside.
    """
    pad = 1e-14 if jnp.finfo(theta.dtype).eps < 1e-10 else 1e-6

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        neg = stationarity(mid, theta, gamma) < 0.0
        return (jnp.where(neg, mid, lo), jnp.where(neg, hi, mid))

    lo = jnp.full(theta.shape[:-1], pad, dtype=theta.dtype)
    hi = jnp.full(theta.shape[:-1], 1.0 - pad, dtype=theta.dtype)
    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
    lam = 0.5 * (lo + hi)
    x = jax.nn.softmax(jnp.sqrt(lam)[..., None] * theta, axis=-1)
    return lam, x
