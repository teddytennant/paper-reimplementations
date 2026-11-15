"""GP relevance predictor, Equation (2) and Appendix B.1.

Dense posterior and a banded fit that never forms a matrix inverse.
Factorization recovery starts at eps = 1e-12, multiplies eps by 10 up to
1e-3, then widens the band when that is allowed.
"""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# Appendix B.1 recovery schedule: 1e-12, 1e-11, ..., 1e-3.
EPS_SCHEDULE = (
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
)
DEFAULT_BANDWIDTH = 128
W_MAX = 1024


@dataclass(frozen=True)
class GPFit:
    """Posterior coefficients for mu(t) = mu_obs + k(t, O_window) dot alpha_window."""

    obs_times: jnp.ndarray
    alpha: jnp.ndarray
    mu_obs: float
    length_scale: float
    eps: float
    bandwidth: int
    widened: bool
    eps_raised: bool
    attempts: tuple


def minmax_normalize(scores):
    """Map scores to [0, 1]. Identical scores become zero, as in Appendix C."""
    values = jnp.asarray(scores, dtype=jnp.float64).reshape(-1)
    if values.size == 0:
        return values
    lo = jnp.min(values)
    hi = jnp.max(values)
    if not bool(jnp.isfinite(lo) and jnp.isfinite(hi)):
        raise ValueError("scores must be finite")
    if float(hi) == float(lo):
        return jnp.zeros_like(values)
    return (values - lo) / (hi - lo)


def rbf_kernel(t, t_prime, length_scale):
    """RBF kernel k(t, t') = exp(-0.5 * ((t - t') / ell)^2)."""
    ell = float(length_scale)
    if not np.isfinite(ell) or ell <= 0.0:
        raise ValueError("length_scale must be positive")
    left = jnp.asarray(t, dtype=jnp.float64)
    right = jnp.asarray(t_prime, dtype=jnp.float64)
    return jnp.exp(-0.5 * jnp.square((left - right) / ell))


def _observation_arrays(timestamps, scores):
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if times.shape != values.shape:
        raise ValueError("timestamps and scores must have the same length")
    if times.size == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("timestamps and scores must be finite")
    if times.size >= 2 and np.any(np.diff(times) <= 0.0):
        raise ValueError("observation timestamps must be strictly increasing")
    return times, values


def _lower_band(times, length_scale, bandwidth, eps):
    """Lower banded storage of K_band + eps I. ab[k, j] = A[j + k, j]."""
    n = times.shape[0]
    width = int(bandwidth)
    ab = np.zeros((width + 1, n), dtype=np.float64)
    ab[0, :] = 1.0 + float(eps)
    if width == 0:
        return ab
    ell = float(length_scale)
    obs = jnp.asarray(times, dtype=jnp.float64)
    for lag in range(1, width + 1):
        dt = obs[lag:] - obs[: n - lag]
        ab[lag, : n - lag] = np.asarray(
            jnp.exp(-0.5 * jnp.square(dt / ell)), dtype=np.float64
        )
    return ab


def _banded_cholesky_solve(ab, rhs):
    """Solve a symmetric banded system by Cholesky. No inverse is formed.

    ab is lower-banded with ab[0] the diagonal. Raises LinAlgError when a
    pivot is not positive.
    """
    width_plus, n = ab.shape
    width = width_plus - 1
    factor = np.array(ab, dtype=np.float64, copy=True)
    target = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if target.shape[0] != n:
        raise ValueError("rhs length must match the banded system")
    for i in range(n):
        j_start = max(0, i - width)
        for j in range(j_start, i):
            acc = factor[i - j, j]
            k_start = max(j_start, j - width)
            for k in range(k_start, j):
                acc -= factor[i - k, k] * factor[j - k, k]
            diag = factor[0, j]
            if not np.isfinite(diag) or diag <= 0.0:
                raise np.linalg.LinAlgError("nonpositive pivot")
            factor[i - j, j] = acc / diag
        acc = factor[0, i]
        for k in range(j_start, i):
            acc -= factor[i - k, k] ** 2
        if not np.isfinite(acc) or acc <= 0.0:
            raise np.linalg.LinAlgError("nonpositive pivot")
        factor[0, i] = np.sqrt(acc)
    forward = np.empty(n, dtype=np.float64)
    for i in range(n):
        acc = target[i]
        for k in range(max(0, i - width), i):
            acc -= factor[i - k, k] * forward[k]
        forward[i] = acc / factor[0, i]
    solution = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        acc = forward[i]
        for k in range(i + 1, min(n, i + width + 1)):
            acc -= factor[k - i, i] * solution[k]
        solution[i] = acc / factor[0, i]
    if not np.all(np.isfinite(solution)):
        raise np.linalg.LinAlgError("non-finite banded solve")
    return solution


def _dense_solve(times, rhs, length_scale, eps):
    """Dense Cholesky solve of (K + eps I) alpha = rhs. No inverse is formed."""
    obs = jnp.asarray(times, dtype=jnp.float64)
    target = jnp.asarray(rhs, dtype=jnp.float64)
    delta = obs[:, None] - obs[None, :]
    kernel = jnp.exp(-0.5 * jnp.square(delta / float(length_scale)))
    system = kernel + float(eps) * jnp.eye(obs.shape[0], dtype=jnp.float64)
    factor = jnp.linalg.cholesky(system)
    if not bool(jnp.all(jnp.isfinite(factor))):
        raise np.linalg.LinAlgError("dense Cholesky failed")
    alpha = jax.scipy.linalg.cho_solve((factor, True), target)
    if not bool(jnp.all(jnp.isfinite(alpha))):
        raise np.linalg.LinAlgError("dense solve failed")
    return np.asarray(alpha, dtype=np.float64)


def fit_gp(
    timestamps,
    scores,
    length_scale,
    bandwidth=None,
    dense=False,
    allow_widen=True,
    w_max=W_MAX,
):
    """Fit the relevance GP on normalized observations.

    mu_obs is the mean of the supplied scores. Pass min-max normalized scores
    when that is the scale used for acquisition. Banded fits keep K_ij only
    for |i - j| <= w. Recovery follows Appendix B.1 and stops at the first
    successful factorization.
    """
    times, values = _observation_arrays(timestamps, scores)
    ell = float(length_scale)
    if not np.isfinite(ell) or ell <= 0.0:
        raise ValueError("length_scale must be positive")
    n = int(times.shape[0])
    mu_obs = float(np.mean(values))
    rhs = values - mu_obs
    cap = max(0, n - 1)
    limit = min(int(w_max), cap)
    if dense:
        initial_w = cap
    elif bandwidth is None:
        initial_w = min(DEFAULT_BANDWIDTH, cap)
    else:
        initial_w = min(max(int(bandwidth), 0), limit)
    attempts = []
    width = initial_w
    while True:
        for eps in EPS_SCHEDULE:
            try:
                if dense:
                    alpha = _dense_solve(times, rhs, ell, eps)
                else:
                    band = _lower_band(times, ell, width, eps)
                    alpha = _banded_cholesky_solve(band, rhs)
            except np.linalg.LinAlgError:
                attempts.append((int(width), float(eps), False))
                continue
            attempts.append((int(width), float(eps), True))
            return GPFit(
                obs_times=jnp.asarray(times, dtype=jnp.float64),
                alpha=jnp.asarray(alpha, dtype=jnp.float64),
                mu_obs=mu_obs,
                length_scale=ell,
                eps=float(eps),
                bandwidth=int(cap if dense else width),
                widened=bool(width > initial_w),
                eps_raised=bool(eps > EPS_SCHEDULE[0]),
                attempts=tuple(attempts),
            )
        if dense or not allow_widen:
            break
        widened = min(2 * width, limit)
        if widened <= width:
            break
        width = widened
    raise np.linalg.LinAlgError(
        "GP factorization failed at the maximum bandwidth and jitter"
    )


def predict(fit, queries):
    """Posterior mean at query timestamps.

    With bandwidth large enough to be dense, the sum uses every observation.
    Otherwise it uses at most 2w + 1 observations around the insertion index.
    A query with no nearby observations returns a value near mu_obs.
    """
    query = np.asarray(queries, dtype=np.float64).reshape(-1)
    obs = np.asarray(fit.obs_times, dtype=np.float64)
    alpha = np.asarray(fit.alpha, dtype=np.float64)
    n = int(obs.shape[0])
    width = int(fit.bandwidth)
    ell = float(fit.length_scale)
    mu_obs = float(fit.mu_obs)
    if query.size == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    if n == 0:
        return jnp.full(query.shape, mu_obs, dtype=jnp.float64)
    query_j = jnp.asarray(query, dtype=jnp.float64)
    if width >= n - 1 or 2 * width + 1 >= n:
        kernel = rbf_kernel(query_j[:, None], jnp.asarray(obs)[None, :], ell)
        mean = mu_obs + kernel @ jnp.asarray(alpha, dtype=jnp.float64)
        return mean
    insertion = np.searchsorted(obs, query, side="left")
    offsets = np.arange(-width, width + 1, dtype=np.int64)
    positions = insertion[:, None] + offsets[None, :]
    positions = positions - np.minimum(positions[:, :1], 0)
    overflow = np.maximum(positions[:, -1:] - (n - 1), 0)
    positions = positions - overflow
    window_times = jnp.asarray(obs[positions], dtype=jnp.float64)
    window_alpha = jnp.asarray(alpha[positions], dtype=jnp.float64)
    kernel = rbf_kernel(query_j[:, None], window_times, ell)
    mean = mu_obs + jnp.sum(kernel * window_alpha, axis=1)
    return mean
