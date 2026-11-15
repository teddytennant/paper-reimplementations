"""The lifted entropic regularizer and its choice map, Section 3.1.

The lifted simplex of eq. (15) is K = {z >= 0 : <1, z> <= 1}, and every nonzero
z in K factors uniquely as z = lambda x with lambda = <1, z> in (0, 1] and x on
the simplex X_K. On that lifting the paper puts the regularizer of eq. (18),

    psi(z) = (lambda + sqrt(lambda)) h(x) - 3 L sqrt(lambda (1 - lambda)),
    L = 4 + log K + (1/4) log^2 K,                                   eq. (17)

with h the negative entropy of eq. (16). The lifted mirror map is Z(y) =
argmax_{z in K} {<y, z> - psi(z)}, eq. (19), and the choice map is its
normalization Q(y) = Z(y) / <1, Z(y)>, eq. (20).

Given the mass lambda the maximization over x is entropic and explicit, which
is eq. (21):

    Q(y) = softmax(sqrt(lambda) / (1 + sqrt(lambda)) * y).

Substituting that maximizer back leaves a one-dimensional concave problem in
lambda over [0, 1], eq. (21)'s "easy one-dimensional problem":

    F(lambda) = (lambda + sqrt(lambda)) * logsumexp(c(lambda) y)
                + 3 L sqrt(lambda (1 - lambda)),   c(lambda) = sqrt(lambda) / (1 + sqrt(lambda)).

F is the partial maximization of a concave function over the slice <1, z> =
lambda, so it is concave in lambda; F'(0+) = +inf and F'(1-) = -inf because of
the sqrt(lambda (1 - lambda)) term, so the maximizer is interior. We solve it
with golden-section search, which needs no derivatives and cannot leave the
bracket. ``tests/test_choice.py`` checks the answer against a dense grid.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0

#: Golden-section iterations. The bracket shrinks by _INV_PHI each step, so 60
#: steps take [0, 1] down to about 3e-13, at the float64 noise floor of F.
DEFAULT_ITERS = 60


def regularizer_constant(num_actions: int) -> float:
    """L = 4 + log K + (1/4) log^2 K, eq. (17) and eq. (25)."""
    log_k = math.log(num_actions)
    return 4.0 + log_k + 0.25 * log_k * log_k


def temperature(lam: jnp.ndarray) -> jnp.ndarray:
    """The lambda-dependent softmax temperature of eq. (21)."""
    root = jnp.sqrt(lam)
    return root / (1.0 + root)


def lifted_value(lam: jnp.ndarray, y: jnp.ndarray, L: float) -> jnp.ndarray:
    """F(lambda), the value of eq. (19) restricted to the slice <1, z> = lambda.

    This is the one-dimensional objective whose maximizer defines the optimal
    mass in eq. (21).
    """
    root = jnp.sqrt(lam)
    entropic = (lam + root) * logsumexp(temperature(lam) * y)
    return entropic + 3.0 * L * jnp.sqrt(jnp.clip(lam * (1.0 - lam), 0.0, None))


def solve_lambda(y: jnp.ndarray, L: float, iters: int = DEFAULT_ITERS) -> jnp.ndarray:
    """Maximize F over lambda in [0, 1] by golden-section search, eq. (21).

    Args:
        y: score vector of shape (K,).
        L: the constant of eq. (17).
        iters: golden-section iterations.

    Returns:
        Scalar optimal mass lambda in (0, 1).
    """

    def body(_, bracket):
        lo, hi = bracket
        span = hi - lo
        left = hi - _INV_PHI * span
        right = lo + _INV_PHI * span
        keep_left = lifted_value(left, y, L) > lifted_value(right, y, L)
        return (jnp.where(keep_left, lo, left), jnp.where(keep_left, right, hi))

    lo, hi = jax.lax.fori_loop(0, iters, body, (jnp.float64(0.0), jnp.float64(1.0)))
    return 0.5 * (lo + hi)


def lifted_choice(y: jnp.ndarray, L: float, iters: int = DEFAULT_ITERS) -> jnp.ndarray:
    """The choice map Q of eq. (20), evaluated through eq. (21).

    Args:
        y: score vector of shape (K,). Algorithm 1 calls this with eta * y.
        L: the constant of eq. (17).
        iters: golden-section iterations for the mass.

    Returns:
        Mixed strategy of shape (K,) on the simplex.
    """
    lam = solve_lambda(y, L, iters)
    return jax.nn.softmax(temperature(lam) * y)


def entropic_choice(y: jnp.ndarray, L: float = 0.0) -> jnp.ndarray:
    """Plain entropic choice map softmax(y), the unlifted baseline.

    This is what ordinary optimistic multiplicative weights uses. ``L`` is
    accepted and ignored so it can stand in for :func:`lifted_choice`.
    """
    del L
    return jax.nn.softmax(y)
