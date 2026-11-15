"""Constitutive laws for the convection term U(L, Y, t) of eq (1).

Everything in this module is swappable: the solver only asks a kinetic model for
(U1, U2) given cell coordinates, the supersaturation S and the temperature T.
Lengths are micrometres, time is seconds, T is degrees Celsius unless a docstring
says otherwise, and rates are um/s.

Models implemented:
  M1               size independent dissolution, eqs (9) and (10)
  M4               size dependent dissolution, eqs (11) and (12)
  SizeDependentGrowth  G1/G2 from Ochsenbein et al., eqs (13) and (14)
  GrowthDissolution    M4 below S = 1, growth above it, as in section 4.3.2
  ConstantRate     a fixed velocity, used to check the scheme against the
                   analytic translation solution
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

# Physical constants from eq (2).
RHO_C = 1.59e-12  # crystal density [g um^-3]
K_V = jnp.pi / 4.0  # shape factor [-]

# Validity ranges, section 4.3. M1 and M4 were fitted for S >= 0.975, the growth
# law for S <= 1.25.
S_MIN = 0.975
S_MAX = 1.25

# Operating limits used in section 4.6.
T_MIN_C = 20.0
T_MAX_C = 41.0
DTDT_MAX = 0.05 / 60.0  # 0.05 C/min expressed in C/s


def solubility(T):
    """Solubility c*(T) [g kg^-1] of L-glutamic acid in water, eq (8). T in C."""
    return 3.37 * jnp.exp(0.036 * T)


def supersaturation(c, T):
    """Relative supersaturation S = c / c*(T).

    Eq (7) as printed reads (c - c*)/c*, but every other part of the paper uses
    S = 1 at equilibrium: M1 (eq 9) has zero rate at S = 1, the growth law needs
    ln(S) > 0 to be real, and the text says "if S = 1 the system is at
    equilibrium". Eq (7) as written is the same quantity minus one, so I use the
    ratio, which is what eqs (9) to (14) and the Smin/Smax bounds require.
    """
    return c / solubility(T)


def concentration_from_S(S, T):
    """Inverse of supersaturation, handy for setting up a run at a target S."""
    return S * solubility(T)


class ConstantRate(eqx.Module):
    """U = (u1, u2) everywhere. Used for the analytic translation checks."""

    u1: jnp.ndarray
    u2: jnp.ndarray

    def __init__(self, u1, u2):
        self.u1 = jnp.asarray(u1, dtype=float)
        self.u2 = jnp.asarray(u2, dtype=float)

    def __call__(self, L1, L2, S, T):
        return jnp.broadcast_to(self.u1, jnp.shape(L1)), jnp.broadcast_to(self.u2, jnp.shape(L2))


class M1(eqx.Module):
    """Size independent linear dissolution, eqs (9) and (10).

        D1 = -k1 (1 - S),  D2 = -k2 (1 - S)

    with the published coefficients k1 = 2.32, k2 = 0.19 um/s. This is the paper's
    predictive baseline and the law I use to generate synthetic ground truth.
    """

    k1: jnp.ndarray
    k2: jnp.ndarray

    def __init__(self, k1=2.32, k2=0.19):
        self.k1 = jnp.asarray(k1, dtype=float)
        self.k2 = jnp.asarray(k2, dtype=float)

    def rates(self, S):
        return -self.k1 * (1.0 - S), -self.k2 * (1.0 - S)

    def __call__(self, L1, L2, S, T):
        d1, d2 = self.rates(S)
        return jnp.broadcast_to(d1, jnp.shape(L1)), jnp.broadcast_to(d2, jnp.shape(L2))


class M4(eqx.Module):
    """Size dependent non-linear dissolution, eqs (11) and (12).

        D1 = -2.24 (1 - S) (1 + L1)^(-5.05e-2)
        D2 = -6.43e-4 (1 - S) (1 + L2)^1.29
    """

    a1: jnp.ndarray
    b1: jnp.ndarray
    a2: jnp.ndarray
    b2: jnp.ndarray

    def __init__(self, a1=2.24, b1=-5.05e-2, a2=6.43e-4, b2=1.29):
        self.a1 = jnp.asarray(a1, dtype=float)
        self.b1 = jnp.asarray(b1, dtype=float)
        self.a2 = jnp.asarray(a2, dtype=float)
        self.b2 = jnp.asarray(b2, dtype=float)

    def __call__(self, L1, L2, S, T):
        d1 = -self.a1 * (1.0 - S) * (1.0 + L1) ** self.b1
        d2 = -self.a2 * (1.0 - S) * (1.0 + L2) ** self.b2
        return d1, d2


class SizeDependentGrowth(eqx.Module):
    """Growth law of Ochsenbein et al., eqs (13) and (14).

        G1 = exp(-6.9e4 / (T^2 ln S)) exp(-1.9e3 / T) (S-1)^(2/3) (ln S)^(1/6) L1^1.2
        G2 = (S-1)^2.7 exp(-7e2 / T)

    The paper lists T in Celsius throughout, but these exponentials only give
    physical rates with T in Kelvin (exp(-700/28) is 1.4e-11 um/s, exp(-700/301)
    is 0.098 um/s), so T is converted to Kelvin here. Both rates are zero for
    S <= 1, and the double-where keeps the reverse mode gradient finite there.
    """

    def __call__(self, L1, L2, S, T):
        Tk = T + 273.15
        growing = S > 1.0
        Ss = jnp.where(growing, S, 2.0)
        lnS = jnp.log(Ss)
        g1 = (
            jnp.exp(-6.9e4 / (Tk**2 * lnS))
            * jnp.exp(-1.9e3 / Tk)
            * (Ss - 1.0) ** (2.0 / 3.0)
            * lnS ** (1.0 / 6.0)
            * L1**1.2
        )
        g2 = (Ss - 1.0) ** 2.7 * jnp.exp(-7e2 / Tk)
        g1 = jnp.broadcast_to(g1, jnp.shape(L1))
        g2 = jnp.broadcast_to(g2, jnp.shape(L2))
        return jnp.where(growing, g1, 0.0), jnp.where(growing, g2, 0.0)


class GrowthDissolution(eqx.Module):
    """M4 for S < 1 and the G1/G2 growth law for S > 1, section 4.3.2.

    This is the combination used for the aspect ratio task. The size dependence
    lives in L1 for growth (the L1^1.2 factor) and in L2 for dissolution, which is
    the asymmetry the optimiser exploits in section 2.3.2.
    """

    dissolution: M4
    growth: SizeDependentGrowth

    def __init__(self, dissolution=None, growth=None):
        self.dissolution = M4() if dissolution is None else dissolution
        self.growth = SizeDependentGrowth() if growth is None else growth

    def __call__(self, L1, L2, S, T):
        d1, d2 = self.dissolution(L1, L2, S, T)
        g1, g2 = self.growth(L1, L2, S, T)
        dissolving = S < 1.0
        u1 = jnp.where(dissolving, d1, g1)
        u2 = jnp.where(dissolving, d2, g2)
        return u1, u2


def supersaturation_penalty(S, s_min=S_MIN, s_max=None):
    """Soft penalty P_S of eq (20): squared violation of the validity range."""
    below = jax.nn.relu(s_min - S) ** 2
    above = 0.0 if s_max is None else jax.nn.relu(S - s_max) ** 2
    return jnp.mean(below + above)


def ramp_rate_penalty(T_values, times, limit=DTDT_MAX):
    """Soft penalty P_Tdot of eq (20) on |dT/dt| exceeding the vessel limit.

    The excess is measured relative to the limit rather than in C/s. In C/s the
    limit is 8.3e-4, so a raw squared violation would be around 1e-4 next to an
    objective of order 1 and the constraint would never bind.
    """
    dT = jnp.diff(T_values) / jnp.diff(times)
    return jnp.mean(jax.nn.relu(jnp.abs(dT) / limit - 1.0) ** 2)
