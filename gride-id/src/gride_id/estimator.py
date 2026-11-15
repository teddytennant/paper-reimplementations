"""Gride distance KL, generic-order MLE, and Full / Profiled calibration.

Equations follow arXiv:2609.37114, Section 3 and supplement S1.1.
Neighbor orders are k1 = k and k2 = 2k. The distance slot of the
calibration compares the data MLE to the integer candidate m with the
closed-form KL. It does not simulate a reference distance MLE.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import betaln, digamma, i0, i1

jax.config.update("jax_enable_x64", True)

_DTYPE = jnp.float64


def i_gamma(gamma, k1, k2) -> float:
    """Finite sum for I(gamma), supplement S1.9.

    n = k2 - k1 - 1 and B is the beta function. The sum is

        I(gamma) = 1/B(k2-k1, k1)
            * sum_{j=0}^{n} (-1)^j * C(n, j) / (k1+j)
            * (digamma(1) - digamma((k1+j)/gamma)).

    When n = 0 the sum is the single j = 0 term. This equals
    E[log((1+Z)^gamma - 1)] for Z ~ BetaPrime(k2-k1, k1), which S1.6
    splits as gamma * E[-log T] + E[log(1 - T^gamma)] for
    T ~ Beta(k1, k2-k1). The second expectation alone is not I(gamma).
    """
    k1 = int(k1)
    k2 = int(k2)
    if k2 <= k1:
        raise ValueError(f"need k1 < k2, got k1={k1}, k2={k2}")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be positive and finite, got {gamma}")
    n = k2 - k1 - 1
    log_b = float(betaln(_DTYPE(k2 - k1), _DTYPE(k1)))
    total = 0.0
    for j in range(n + 1):
        s = k1 + j
        coef = ((-1) ** j) * math.comb(n, j) / s
        psi_gap = float(digamma(1.0) - digamma(s / gamma))
        total += coef * psi_gap
    return math.exp(-log_b) * total


def gride_kl(d_dist, d_ref, k1, k2) -> float:
    """Closed-form KL between distance laws at d_dist and d_ref (S1.10).

    gamma = d_ref / d_dist. When d_dist == d_ref the value is exactly 0.
    """
    k1 = int(k1)
    k2 = int(k2)
    if k2 <= k1:
        raise ValueError(f"need k1 < k2, got k1={k1}, k2={k2}")
    d_dist = float(d_dist)
    d_ref = float(d_ref)
    if d_dist == d_ref:
        return 0.0
    if d_dist <= 0.0 or d_ref <= 0.0:
        raise ValueError("d_dist and d_ref must be positive")
    gamma = d_ref / d_dist
    ig = i_gamma(gamma, k1, k2)
    kl = (
        math.log(d_dist / d_ref)
        + (k2 - 1) * (gamma - 1.0) * float(digamma(k2) - digamma(k1))
        + (k2 - k1 - 1) * float(digamma(k2 - k1) - digamma(k1) - ig)
    )
    return float(kl)


def _mean_score(d, log_mu, k1, k2):
    """Derivative of the mean log density in d.

    log f = log d + (k2-k1-1) log(mu^d - 1)
            - (d*(k2-1)+1) log mu - log B(k2-k1, k1).
    mu^d / (mu^d - 1) = -1 / expm1(-d log mu), which stays finite for
    large d.
    """
    dlog = d * log_mu
    ratio = -1.0 / jnp.expm1(-dlog)
    n_extra = k2 - k1 - 1
    mean_log = jnp.mean(log_mu)
    return 1.0 / d + n_extra * jnp.mean(ratio * log_mu) - (k2 - 1) * mean_log


def gride_mle(mus, k1, k2) -> float:
    """Maximize sum log f(mu_i; d, k1, k2) over d > 0.

    For k1 = 1, k2 = 2 the density is d / mu^(d+1) and the maximizer is
    N / sum(log mu). Other orders use bisection on the score. The score
    is strictly decreasing, so the root is the unique MLE.
    """
    k1 = int(k1)
    k2 = int(k2)
    if k2 <= k1:
        raise ValueError(f"need k1 < k2, got k1={k1}, k2={k2}")
    mus = jnp.asarray(mus, dtype=_DTYPE).reshape(-1)
    if mus.shape[0] == 0:
        raise ValueError("mus must be non-empty")
    # Ratios at or below 1 are outside the support. A tiny lift keeps
    # the score finite if a neighbor tie underflows to 1.
    mus = jnp.maximum(mus, 1.0 + 1e-12)
    log_mu = jnp.log(mus)
    if k1 == 1 and k2 == 2:
        return float(mus.shape[0] / jnp.sum(log_mu))

    lo = _DTYPE(1e-6)
    hi = _DTYPE(1e3)
    for _ in range(30):
        if float(_mean_score(hi, log_mu, k1, k2)) <= 0.0:
            break
        hi = hi * 2.0
    for _ in range(30):
        if float(_mean_score(lo, log_mu, k1, k2)) >= 0.0:
            break
        lo = lo * 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        score = _mean_score(mid, log_mu, k1, k2)
        lo = jnp.where(score > 0.0, mid, lo)
        hi = jnp.where(score > 0.0, hi, mid)
    return float(0.5 * (lo + hi))


def _squared_distances(x):
    diff = x[:, None, :] - x[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def pairwise_ratios(x, k1, k2):
    """Euclidean neighbor ratios mu_i = r_{k2} / r_{k1}, self excluded."""
    k1 = int(k1)
    k2 = int(k2)
    if k1 < 1 or k2 <= k1:
        raise ValueError(f"need 1 <= k1 < k2, got k1={k1}, k2={k2}")
    x = jnp.asarray(x, dtype=_DTYPE)
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, D), got {tuple(x.shape)}")
    n = int(x.shape[0])
    if n <= k2:
        raise ValueError(f"need more than k2={k2} points, got N={n}")
    dist2 = jnp.maximum(_squared_distances(x), 0.0)
    dist2 = dist2.at[jnp.diag_indices(n)].set(jnp.inf)
    order = jnp.argsort(dist2, axis=1)
    rows = jnp.arange(n)
    r1 = jnp.sqrt(dist2[rows, order[:, k1 - 1]])
    r2 = jnp.sqrt(dist2[rows, order[:, k2 - 1]])
    return r2 / jnp.maximum(r1, 1e-15)


def two_nn_angles(x):
    """Angle between the vectors from each point to its two nearest neighbors.

    These are the two-nearest-neighbor angles, not the k and 2k neighbors
    used for the distance ratios.
    """
    x = jnp.asarray(x, dtype=_DTYPE)
    n = int(x.shape[0])
    if n < 3:
        raise ValueError(f"two-NN angles need at least 3 points, got N={n}")
    dist2 = jnp.maximum(_squared_distances(x), 0.0)
    dist2 = dist2.at[jnp.diag_indices(n)].set(jnp.inf)
    nearest = jnp.argsort(dist2, axis=1)[:, :2]
    v1 = x[nearest[:, 0]] - x
    v2 = x[nearest[:, 1]] - x
    dot = jnp.sum(v1 * v2, axis=1)
    n1 = jnp.linalg.norm(v1, axis=1)
    n2 = jnp.linalg.norm(v2, axis=1)
    cos = jnp.clip(dot / jnp.maximum(n1 * n2, 1e-15), -1.0, 1.0)
    return jnp.arccos(cos)


def fisher_concentration(resultant):
    """Fisher piecewise approximation of von Mises concentration.

    The paper names this approximation and does not write it. R is the
    mean resultant length. If R is within 1e-8 of 0, tau is 0. Otherwise
    R is clipped to [0, 1-1e-8] before the last piece:

        R < 0.53:  tau = 2R + R^3 + 5 R^5 / 6
        R < 0.85:  tau = -0.4 + 1.39 R + 0.43 / (1 - R)
        else:      tau = 1 / (R^3 - 4 R^2 + 3 R)
    """
    r = float(resultant)
    if r <= 1e-8:
        return 0.0
    r = min(max(r, 0.0), 1.0 - 1e-8)
    if r < 0.53:
        return 2.0 * r + r**3 + 5.0 * r**5 / 6.0
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    return 1.0 / (r**3 - 4.0 * r**2 + 3.0 * r)


def von_mises_fit(angles):
    """Mean direction and Fisher concentration of angles on the circle."""
    angles = jnp.asarray(angles, dtype=_DTYPE).reshape(-1)
    if angles.shape[0] == 0:
        raise ValueError("angles must be non-empty")
    sin_mean = jnp.mean(jnp.sin(angles))
    cos_mean = jnp.mean(jnp.cos(angles))
    resultant = float(jnp.sqrt(sin_mean**2 + cos_mean**2))
    nu = float(jnp.arctan2(sin_mean, cos_mean))
    tau = fisher_concentration(resultant)
    return nu, tau


def _bessel_ratio(tau):
    """A(tau) = I1(tau) / I0(tau), with A(0) = 0."""
    tau = float(tau)
    if tau == 0.0:
        return 0.0
    return float(i1(tau) / i0(tau))


def von_mises_kl(nu1, tau1, nu2, tau2, profiled: bool) -> float:
    """Von Mises KL. Profiled replaces nu2 by nu1 before the KL.

    KL = log(I0(tau2) / I0(tau1)) + A(tau1) * (tau1 - tau2 * cos(nu1 - nu2)).
    At tau1 = tau2 = 0 this is 0. Profiled ignores a pure location shift.
    """
    nu1 = float(nu1)
    tau1 = float(tau1)
    nu2 = float(nu2)
    tau2 = float(tau2)
    if profiled:
        nu2 = nu1
    if tau1 == 0.0 and tau2 == 0.0:
        return 0.0
    log_term = float(jnp.log(i0(tau2)) - jnp.log(i0(tau1)))
    kl = log_term + _bessel_ratio(tau1) * (tau1 - tau2 * math.cos(nu1 - nu2))
    return float(kl)


def _sample_unit_ball(key, n_ref, dimension):
    """n_ref points uniform in the unit ball of R^dimension.

    Direction is a normalized Gaussian. Radius is U^{1/dimension} with
    U uniform on (0, 1).
    """
    key_dir, key_rad = jax.random.split(key)
    gauss = jax.random.normal(key_dir, (n_ref, dimension), dtype=_DTYPE)
    norms = jnp.linalg.norm(gauss, axis=1, keepdims=True)
    direction = gauss / jnp.maximum(norms, 1e-12)
    radius = jax.random.uniform(key_rad, (n_ref, 1), dtype=_DTYPE) ** (
        1.0 / dimension
    )
    return direction * radius


def estimate_id(x, *, k=5, m_max=None, n_ref=256, key, profiled=False) -> dict:
    """Componentwise Gride estimate on a point cloud of shape (N, D).

    Distance ratios use orders k1 = k and k2 = 2k. Angular statistics use
    two-nearest-neighbor angles. Candidate integers are m = 1..m_max.
    J(m) = KL(d_hat_dist, m) + angular KL against two-NN angles of n_ref
    uniform unit-ball points in R^m. Full uses the von Mises KL. Profiled
    aligns the reference mean direction with the data direction first.
    The returned estimate is argmin_m J(m).

    m_max defaults to min(D, 10) and is never below 1. Raises ValueError
    if N < k2 + 2.
    """
    if key is None:
        raise ValueError("key is required")
    x = jnp.asarray(x, dtype=_DTYPE)
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, D), got {tuple(x.shape)}")
    n, ambient = int(x.shape[0]), int(x.shape[1])
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    k1 = k
    k2 = 2 * k
    if n < k2 + 2:
        raise ValueError(
            f"need at least k2+2={k2 + 2} points for k1={k1}, k2={k2}, got N={n}"
        )
    n_ref = int(n_ref)
    if n_ref < 3:
        raise ValueError(f"n_ref must be at least 3, got {n_ref}")
    if m_max is None:
        m_max = max(1, min(ambient, 10))
    else:
        m_max = int(m_max)
    if m_max < 1:
        raise ValueError(f"m_max must be at least 1, got {m_max}")

    ratios = pairwise_ratios(x, k1, k2)
    d_dist = gride_mle(ratios, k1, k2)
    nu, tau = von_mises_fit(two_nn_angles(x))

    keys = jax.random.split(key, m_max)
    discrepancies = []
    for index, dimension in enumerate(range(1, m_max + 1)):
        reference = _sample_unit_ball(keys[index], n_ref, dimension)
        nu_m, tau_m = von_mises_fit(two_nn_angles(reference))
        kl_dist = gride_kl(d_dist, float(dimension), k1, k2)
        kl_ang = von_mises_kl(nu, tau, nu_m, tau_m, profiled)
        discrepancies.append(kl_dist + kl_ang)
    j = jnp.asarray(discrepancies, dtype=_DTYPE)
    d_hat = int(jnp.argmin(j)) + 1
    return {
        "d_hat": d_hat,
        "d_dist": float(d_dist),
        "j": j,
        "nu": float(nu),
        "tau": float(tau),
    }
