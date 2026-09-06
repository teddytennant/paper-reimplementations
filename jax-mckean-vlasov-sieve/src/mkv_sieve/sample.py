"""Sampling from the stationary density pi_xi of eq (3.4).

Section 3.3: if X ~ pi_xi then S = V(X) has density f_xi(s) = q_xi(s) m_V(s),
and given S = s the law of X on the level set {V = s} does not depend on xi.
So we draw the energy first and then a point on its level set.  For the OU
potential (1.2) the level set is the sphere of radius sqrt(s - 1), so the second
step is a uniform direction.

We draw the radius rather than the energy because the radial density
omega_d rho^{d-1} q_xi(1 + rho^2) is smooth at the minimum while f_xi itself
blows up there for d = 1.  Inversion is linear interpolation of the tabulated
cdf followed by one Newton step against the cubic-Hermite cdf, which puts the
inversion error well under the Monte Carlo noise at the sample sizes here.
"""

import jax
import jax.numpy as jnp

from mkv_sieve.profile import Profile, profile_energy_cdf


def sample_radius(key, prof: Profile, n: int):
    """Draw n radii from the law of ||X|| under pi_xi."""
    u = jax.random.uniform(key, (n,), minval=0.0, maxval=1.0)
    r = jnp.interp(u, prof.cdf, prof.rho)
    f = jnp.interp(r, prof.rho, prof.pdf_rho)
    step = (profile_energy_cdf(prof, 1.0 + r * r) - u) / jnp.where(f > 0.0, f, 1.0)
    h = prof.rho[1] - prof.rho[0]
    step = jnp.where(f > 1e-10, jnp.clip(step, -0.5 * h, 0.5 * h), 0.0)
    return jnp.clip(r - step, prof.rho[0], prof.rho[-1])


def sample_energies(key, prof: Profile, n: int):
    """Draw n energies S = V(X) with X ~ pi_xi, density f_xi(s) = q_xi(s) m_V(s)."""
    r = sample_radius(key, prof, n)
    return prof.v_star + r * r


def sample_particles(key, prof: Profile, pot, n: int):
    """Draw n particles X ~ pi_xi in R^d, eq (3.4)."""
    k_r, k_dir = jax.random.split(key)
    r = sample_radius(k_r, prof, n)
    z = jax.random.normal(k_dir, (n, pot.d))
    direction = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    return r[:, None] * direction
