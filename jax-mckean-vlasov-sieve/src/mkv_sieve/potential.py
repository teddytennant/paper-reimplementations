"""The potential V and the level-set measure m_V of Assumption 3.1.

Only the Ornstein-Uhlenbeck potential of eq (1.2) is implemented,

    V(x) = 1 + ||x||_2^2,

for which Remark 3.2 gives v_star = 1 and m_V(s) = (omega_d / 2) (s - 1)^{d/2 - 1}
with omega_d = 2 pi^{d/2} / Gamma(d/2).

Everything downstream talks to a potential through four things: v_star, V, the
radial weight omega_d rho^{d-1} that turns a level-set integral into a radial
one, and log Z_c from Remark 3.6.  A different radially symmetric V would only
need those four replaced.
"""

import math
from typing import NamedTuple

import jax.numpy as jnp


def sphere_area(d: int) -> float:
    """omega_d = 2 pi^{d/2} / Gamma(d/2), the surface area of the unit sphere in R^d.

    Plain Python, not jnp, so it stays a compile-time constant under jit.
    """
    return 2.0 * math.exp(0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d))


class OUPotential(NamedTuple):
    """V(x) = 1 + ||x||^2 on R^d, eq (1.2)."""

    d: int

    @property
    def v_star(self) -> float:
        return 1.0

    def V(self, x):
        """V(x) = 1 + ||x||_2^2.  Batched over all leading axes of x."""
        return 1.0 + jnp.sum(x * x, axis=-1)

    def grad_V(self, x):
        return 2.0 * x

    def m_V(self, s):
        """Level-set measure density, Remark 3.2."""
        return 0.5 * sphere_area(self.d) * jnp.power(s - 1.0, 0.5 * self.d - 1.0)

    def radius(self, s):
        """The radius of the level set {V = s}."""
        return jnp.sqrt(jnp.maximum(s - 1.0, 0.0))

    def energy(self, rho):
        """V at radius rho."""
        return 1.0 + rho * rho

    def radial_weight(self, rho):
        """The factor w with int_{R^d} h(V(x)) dx = int_0^inf h(V(rho)) w(rho) drho.

        For a radial V this is omega_d rho^{d-1}, and it equals m_V(s) ds under
        s = 1 + rho^2.  Using rho rather than s keeps the integrand smooth at the
        minimum, where m_V itself blows up for d = 1.
        """
        return sphere_area(self.d) * jnp.power(rho, self.d - 1.0)

    def log_Z(self, c: float) -> float:
        """log of Z_c = int exp(-c (V(x) - v_star)) dx from Remark 3.6.

        For the OU potential Z_c = (pi / c)^{d/2}, so a_xi is pinned between
        (kappa/pi)^{d/2} and (K/pi)^{d/2}.
        """
        return 0.5 * self.d * (math.log(math.pi) - math.log(c))
