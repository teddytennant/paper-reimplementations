"""Modified Bessel function of the third kind, on the log scale, in pure JAX.

The GHST density in Section 2.4 of the paper and the GIG normalising constant in
its equation (13) both need K_nu(x). JAX has no Bessel K, and calling scipy from
inside a jitted function is not an option, so this module computes it directly.

The integral representation

    K_nu(x) = (1/2) integral over the real line of exp(nu*t - x*cosh(t)) dt

has an analytic, doubly-exponentially decaying integrand, so a plain trapezoid
rule converges geometrically once the grid is centred and scaled. The peak of
nu*t - x*cosh(t) sits at t* = asinh(nu/x) and its second derivative there is
-sqrt(x^2 + nu^2), so the grid is placed at t* + s/(x^2+nu^2)^(1/4) for s on a
fixed symmetric range. Summing in log space with logsumexp keeps the huge and
tiny regimes (small x with large nu, or large x) representable.

Against scipy.special.kve this reaches about 1e-13 absolute error on log K over
nu in [0.5, 101] and x in [1e-6, 1e3], which is float64 rounding for values of
that size. See tests/test_bessel.py.

Because the whole thing is a differentiable JAX expression, d/dnu log K_nu(x),
which equation (22) needs for E(log W), comes straight out of jax.grad instead
of a finite difference.
"""

import jax
import jax.numpy as jnp

_N_NODES = 769
_HALF_WIDTH = 30.0


def log_bessel_k(nu, x):
    """log K_nu(x) for real order nu and x > 0.

    K_{-nu} = K_nu, so the order is folded to its absolute value first; the model
    uses lambda = -(nu_s + p)/2 < 0 in Proposition 2.
    """
    nu = jnp.abs(jnp.asarray(nu, dtype=jnp.float64))
    x = jnp.asarray(x, dtype=jnp.float64)
    nu, x = jnp.broadcast_arrays(nu, x)

    t_star = jnp.arcsinh(nu / x)
    scale = (x * x + nu * nu) ** -0.25

    s = jnp.linspace(-_HALF_WIDTH, _HALF_WIDTH, _N_NODES)
    step = s[1] - s[0]
    t = t_star[..., None] + scale[..., None] * s
    g = nu[..., None] * t - x[..., None] * jnp.cosh(t)
    return jax.scipy.special.logsumexp(g, axis=-1) + jnp.log(step * scale) - jnp.log(2.0)


_dlog_k = jax.grad(lambda nu, x: log_bessel_k(nu, x).sum(), argnums=0)


def dlog_bessel_k_dnu(nu, x):
    """d/dnu log K_nu(x), the second term of equation (22).

    Autodiff of the quadrature above. The order is not folded here: the
    derivative at a negative order is the negative of the derivative at |nu|,
    and equation (22) wants it at lambda itself.
    """
    nu = jnp.asarray(nu, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    nu, x = jnp.broadcast_arrays(nu, x)
    return jnp.sign(nu) * _dlog_k(jnp.abs(nu), x)
