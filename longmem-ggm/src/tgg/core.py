"""Direct statistic, block bootstrap, and consistency-empowered correction.

The public test is Appendix Algorithm 1. It compares the unstudentized max
residual (main-text equation for T_hat) to the overlapping block-bootstrap
quantile of the plug-in scores. The appendix algorithm does not studentize,
and neither does this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class TestResult:
    """Outcome of a level-alpha test.

    statistic is T_hat for the direct test and T_tilde for the
    consistency-empowered test. critical_value is the bootstrap quantile
    q_hat. reject is True when statistic > critical_value.
    """

    statistic: float
    critical_value: float
    reject: bool

    @property
    def T_hat(self) -> float:
        return self.statistic

    @property
    def q_hat(self) -> float:
        return self.critical_value


def suggested_block_length(n: int, kind: str = "one_third") -> int:
    """Block length choices used around the paper.

    kind="one_third" returns floor(n^{1/3}), the smoke default.
    kind="two_thirds" returns floor(n^{2/3}), the simulation-section choice.
    The theory rate is l on the order of n^phi log^psi(p), which is not a
    single closed default.
    """

    if n < 1:
        raise ValueError("n must be positive")
    if kind == "one_third":
        exponent = 1.0 / 3.0
    elif kind == "two_thirds":
        exponent = 2.0 / 3.0
    else:
        raise ValueError("kind must be 'one_third' or 'two_thirds'")
    return max(1, int(math.floor(n**exponent)))


def sample_covariance(X) -> jnp.ndarray:
    """Sample covariance with the paper's normalization.

    Sigma_hat = (1/n) sum_{t=1}^n X_t X_t^T.

    This is the normalization in the methodology section, not the unbiased
    1/(n-1) estimator. X has shape (n, p), rows are time points.
    """

    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    n = X.shape[0]
    if n < 1:
        raise ValueError("n must be positive")
    return (X.T @ X) / n


def column_supports(edges, p: int):
    """Neighborhoods S_{0,j} = {j} union {i : (i, j) is an undirected edge}.

    edges is a sequence of 0-based pairs. The graph is treated as undirected.
    The diagonal is always included, matching the paper's definition of S_{0,j}.
    """

    neigh = [set() for _ in range(p)]
    if edges is None:
        pair_arr = np.zeros((0, 2), dtype=np.int64)
    else:
        pair_arr = np.asarray(edges, dtype=np.int64)
        if pair_arr.size == 0:
            pair_arr = np.zeros((0, 2), dtype=np.int64)
        else:
            pair_arr = pair_arr.reshape(-1, 2)
    for i, j in pair_arr:
        i = int(i)
        j = int(j)
        if i == j:
            continue
        if not (0 <= i < p and 0 <= j < p):
            raise ValueError(f"edge {(i, j)} is outside 0..{p - 1}")
        neigh[i].add(j)
        neigh[j].add(i)
    supports = []
    for j in range(p):
        supports.append(np.array(sorted(neigh[j] | {j}), dtype=np.int32))
    return supports


def _invert_or_zero(M, tol: float = 1e-10):
    """Inverse of a symmetric matrix, or the zero matrix if it is singular.

    The paper sets (B^T Sigma B)^{-1} to the zero matrix whenever that
    Gram matrix is singular. The neighborhood is small, so the factorization
    is a NumPy helper; the residual and scores that use it stay in JAX.
    """

    sym = np.asarray(M, dtype=np.float64)
    sym = 0.5 * (sym + sym.T)
    evals, evecs = np.linalg.eigh(sym)
    scale = max(1.0, float(np.max(np.abs(evals))))
    if float(np.min(evals)) <= tol * scale:
        return np.zeros_like(sym), True
    safe = np.where(evals > tol * scale, 1.0 / evals, 0.0)
    return (evecs * safe) @ evecs.T, False


def _fit_local(Sigma, edges):
    """Neighborhood supports, local-inverse columns, and Gram inverses."""

    Sigma_np = np.asarray(jnp.asarray(Sigma), dtype=np.float64)
    p = int(Sigma_np.shape[0])
    supports = column_supports(edges, p)
    omega = np.zeros((p, p), dtype=np.float64)
    invs = []
    singulars = []
    for j, S in enumerate(supports):
        inv, singular = _invert_or_zero(Sigma_np[np.ix_(S, S)])
        invs.append(inv)
        singulars.append(singular)
        if singular:
            continue
        pos = int(np.where(S == j)[0][0])
        omega[S, j] = inv[:, pos]
    return supports, jnp.asarray(omega), invs, singulars, jnp.asarray(Sigma_np)


def local_inverse(Sigma, edges):
    """Local inverse columns omega_hat_{0,j}.

    omega_hat_{0,j} = B (B^T Sigma B)^{-1} B^T e_j,
    where B selects coordinates in S_{0,j}. Columns of the returned (p, p)
    matrix are these vectors. A singular neighborhood Gram matrix yields a
    zero column.
    """

    _supports, omega, _invs, _singulars, _Sigma = _fit_local(Sigma, edges)
    return omega


def _off_support_mask(supports, p: int) -> np.ndarray:
    mask = np.ones((p, p), dtype=bool)
    for j, S in enumerate(supports):
        mask[S, j] = False
    return mask


def _support_mask(supports, p: int) -> np.ndarray:
    mask = np.zeros((p, p), dtype=bool)
    for j, S in enumerate(supports):
        mask[S, j] = True
    return mask


def direct_statistic(Sigma, edges, n: int) -> float:
    """Unstudentized max residual.

    T_hat = sqrt(n) * max_{(i, j) in J_0} |e_i^T Sigma omega_hat_{0,j}|,
    J_0 = {(i, j) : i not in S_{0,j}}.

    Passing the population covariance Sigma = Omega^{-1} with E_0 equal to
    the support of Omega gives T_hat = 0 (up to numerical error), because
    the local inverse recovers the precision columns and off-support
    residuals vanish.
    """

    supports, omega, _invs, _singulars, Sigma = _fit_local(Sigma, edges)
    return _max_residual_statistic(Sigma, omega, supports, n)


def _max_residual_statistic(Sigma, omega, supports, n: int) -> float:
    fitted = Sigma @ omega
    mask = _off_support_mask(supports, fitted.shape[0])
    if not np.any(mask):
        return 0.0
    residual = jnp.max(jnp.abs(fitted[mask]))
    return float(jnp.sqrt(float(n)) * residual)


def plugin_scores(X, Sigma, edges):
    """Plug-in scores, already centered.

    u_hat_ij = e_i - B (B^T Sigma B)^{-1} B^T Sigma e_i
    X_hat_{t,ij} = u_hat_ij^T (X_t X_t^T - Sigma) omega_hat_{0,j}
                 = (u · X_t) (omega · X_t) - u^T Sigma omega

    Returns scores of shape (n, |J_0|), the pair index array of shape
    (|J_0|, 2), and the local-inverse matrix.
    """

    X = jnp.asarray(X, dtype=jnp.float64)
    n, p = int(X.shape[0]), int(X.shape[1])
    supports, omega, invs, singulars, Sigma = _fit_local(Sigma, edges)
    Sigma_np = np.asarray(Sigma)
    pairs_i = []
    pairs_j = []
    U_rows = []
    for j, S in enumerate(supports):
        S_set = set(int(i) for i in S.tolist())
        inv = invs[j]
        singular = singulars[j]
        for i in range(p):
            if i in S_set:
                continue
            u = np.zeros(p, dtype=np.float64)
            if singular:
                u[i] = 1.0
            else:
                u[S] = -(inv @ Sigma_np[S, i])
                u[i] = 1.0
            U_rows.append(u)
            pairs_i.append(i)
            pairs_j.append(j)
    pairs = np.column_stack([pairs_i, pairs_j]) if pairs_i else np.zeros((0, 2), dtype=np.int64)
    if not pairs_i:
        return jnp.zeros((n, 0), dtype=jnp.float64), pairs, omega
    U = jnp.asarray(np.stack(U_rows))
    W = omega[:, np.asarray(pairs_j, dtype=np.int64)].T
    ux = X @ U.T
    wx = X @ W.T
    center = jnp.einsum("mp,pq,mq->m", U, Sigma, W)
    scores = ux * wx - center
    return scores, pairs, omega


def overlapping_block_sums(scores, l: int) -> jnp.ndarray:
    """B_{r,l} = sum_{t=r-l+1}^r scores_t for r = l, ..., n (1-based).

    scores has shape (n, m). The returned array has shape (n - l + 1, m).
    """

    scores = jnp.asarray(scores, dtype=jnp.float64)
    n, m = scores.shape
    if l < 1 or l > n:
        raise ValueError("block length l must satisfy 1 <= l <= n")
    if m == 0:
        return jnp.zeros((n - l + 1, 0), dtype=jnp.float64)
    csum = jnp.cumsum(scores, axis=0)
    total = csum[l - 1 :]
    prev = jnp.concatenate(
        [jnp.zeros((1, m), dtype=jnp.float64), csum[: n - l]],
        axis=0,
    )
    return total - prev


def empirical_bootstrap_quantile(block_stats, alpha: float) -> float:
    """q_hat = inf{u : F_hat(u) >= 1 - alpha}.

    F_hat is the empirical CDF of the block statistics, so q_hat is an
    order statistic, not an interpolated quantile.
    """

    values = np.sort(np.asarray(block_stats, dtype=np.float64).reshape(-1))
    m = int(values.shape[0])
    if m == 0:
        return 0.0
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    # Smallest integer k with k/m >= 1 - alpha.
    k = math.ceil((1.0 - alpha) * m - 1e-12)
    k = max(1, min(m, k))
    return float(values[k - 1])


def block_bootstrap_quantile(scores, l: int, alpha: float) -> float:
    """Quantile of l^{-1/2} ||B_{r,l}||_inf over overlapping blocks."""

    scores = jnp.asarray(scores, dtype=jnp.float64)
    blocks = overlapping_block_sums(scores, l)
    if blocks.shape[1] == 0:
        return 0.0
    inf_norms = jnp.max(jnp.abs(blocks), axis=1) / math.sqrt(l)
    return empirical_bootstrap_quantile(inf_norms, alpha)


def _resolve_l(n: int, l) -> int:
    if l is None:
        return suggested_block_length(n, "one_third")
    l = int(l)
    if l < 1 or l > n:
        raise ValueError("block length l must satisfy 1 <= l <= n")
    return l


def direct_test(X, edges, l=None, alpha: float = 0.05) -> TestResult:
    """Appendix Algorithm 1.

    Reject H_0 when T_hat > q_hat. l and alpha are arguments. The default
    block length is floor(n^{1/3}). The simulation section of the paper uses
    l = n^{2/3}; pass that explicitly if you want it.
    """

    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    n = int(X.shape[0])
    l = _resolve_l(n, l)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    Sigma = sample_covariance(X)
    scores, _pairs, _omega = plugin_scores(X, Sigma, edges)
    statistic = direct_statistic(Sigma, edges, n)
    critical = block_bootstrap_quantile(scores, l, alpha)
    return TestResult(statistic, critical, bool(statistic > critical))


def screen_correction(omega, tau, on_support, C_n: float, delta_n: float):
    """Consistency-empowered pump on the null support.

    For i in S_{0,j},
        Delta_ij = C_n * 1{|omega_ij| / tau_ij <= delta_n},
    and Delta_ij = 0 for i not in S_{0,j}.

    A near-zero on-support entry is increased by C_n. A large on-support
    entry is left unchanged. Off-support entries are never pumped.

    The comparison is implemented as |omega| <= delta_n * tau, which matches
    the ratio form for tau >= 0 and treats tau = 0 without division.
    Returns (omega + Delta, Delta).
    """

    omega = jnp.asarray(omega, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    on_support = jnp.asarray(on_support).astype(bool)
    near_zero = jnp.abs(omega) <= (float(delta_n) * jnp.maximum(tau, 0.0))
    delta = jnp.where(on_support & near_zero, float(C_n), 0.0)
    return omega + delta, delta


def _long_run_tau(X, Sigma, edges, omega, l: int) -> jnp.ndarray:
    """Block-bootstrap long-run standard errors tau_ij = sqrt(L_ij / n).

    For i in S_{0,j},
        v_ij = B (B^T Sigma B)^{-1} B^T e_i,
        X^omega_{t,ij} = v_ij^T (X_t X_t^T - Sigma) omega_j,
        L_ij = average over blocks of (l^{-1/2} sum_{t in block} X^omega_{t,ij})^2.
    Off-support entries of tau are zero and are not used by the screen.
    """

    X = jnp.asarray(X, dtype=jnp.float64)
    n, p = int(X.shape[0]), int(X.shape[1])
    supports, omega_fit, invs, singulars, Sigma = _fit_local(Sigma, edges)
    if omega is None:
        omega = omega_fit
    else:
        omega = jnp.asarray(omega, dtype=jnp.float64)
    pairs_i = []
    pairs_j = []
    V_rows = []
    for j, S in enumerate(supports):
        inv = invs[j]
        singular = singulars[j]
        for pos, i in enumerate(S.tolist()):
            v = np.zeros(p, dtype=np.float64)
            if not singular:
                v[S] = inv[:, pos]
            V_rows.append(v)
            pairs_i.append(int(i))
            pairs_j.append(j)
    tau = jnp.zeros((p, p), dtype=jnp.float64)
    if not pairs_i:
        return tau
    V = jnp.asarray(np.stack(V_rows))
    W = omega[:, np.asarray(pairs_j, dtype=np.int64)].T
    center = jnp.einsum("mp,pq,mq->m", V, Sigma, W)
    scores = (X @ V.T) * (X @ W.T) - center
    blocks = overlapping_block_sums(scores, l)
    scaled = blocks / math.sqrt(l)
    long_run = jnp.mean(scaled**2, axis=0)
    tau_pairs = jnp.sqrt(jnp.maximum(long_run, 0.0) / n)
    tau = tau.at[np.asarray(pairs_i, dtype=np.int64), np.asarray(pairs_j, dtype=np.int64)].set(
        tau_pairs
    )
    return tau


def _ce_tuning(n: int, p: int, C_delta: float, delta_n, C_n):
    """Paper defaults: delta_n = sqrt(log n), C_n = C_delta * sqrt(log p).

    log is the natural logarithm.
    """

    if delta_n is None:
        delta_n = float(math.sqrt(math.log(n)))
    if C_n is None:
        C_n = float(C_delta * math.sqrt(math.log(max(p, 2))))
    return float(delta_n), float(C_n)


def consistency_empowered_statistic(
    X,
    edges,
    l=None,
    C_delta: float = 1.0,
    delta_n=None,
    C_n=None,
    tau=None,
):
    """Corrected statistic T_tilde.

    omega_tilde_{0,j} = omega_hat_{0,j} + Delta_j,
    T_tilde = sqrt(n) * max_{i,j} |e_i^T Sigma omega_tilde_{0,j} - 1{i = j}|.

    If tau is provided it is used as the long-run standard-error matrix and
    the block estimator is skipped. That path is for direct checks of the
    screening formula. Otherwise tau is the block-bootstrap estimator in the
    appendix.

    Under a correct E_0 and an invertible neighborhood Gram matrix, a zero
    pump gives T_tilde = T_hat, because on-support residuals of the local
    inverse are exactly 1{i = j}.
    """

    X = jnp.asarray(X, dtype=jnp.float64)
    n, p = X.shape
    l = _resolve_l(n, l)
    delta_n, C_n = _ce_tuning(n, p, C_delta, delta_n, C_n)
    Sigma = sample_covariance(X)
    supports = column_supports(edges, p)
    omega = local_inverse(Sigma, edges)
    if tau is None:
        tau = _long_run_tau(X, Sigma, edges, omega, l)
    else:
        tau = jnp.asarray(tau, dtype=jnp.float64)
    on_support = _support_mask(supports, p)
    omega_tilde, delta = screen_correction(omega, tau, on_support, C_n, delta_n)
    fitted = Sigma @ omega_tilde
    resid = fitted - jnp.eye(p, dtype=jnp.float64)
    statistic = float(jnp.sqrt(float(n)) * jnp.max(jnp.abs(resid)))
    return statistic, omega_tilde, delta, tau


def ce_test(
    X,
    edges,
    l=None,
    alpha: float = 0.05,
    C_delta: float = 1.0,
    delta_n=None,
    C_n=None,
) -> TestResult:
    """Appendix Algorithm 2.

    The critical value is the same block-bootstrap quantile used by the
    direct test (plug-in scores of the uncorrected local inverse). Reject
    when T_tilde > q_hat.
    """

    X = jnp.asarray(X, dtype=jnp.float64)
    n = int(X.shape[0])
    l = _resolve_l(n, l)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    statistic, _omega_tilde, _delta, _tau = consistency_empowered_statistic(
        X,
        edges,
        l=l,
        C_delta=C_delta,
        delta_n=delta_n,
        C_n=C_n,
    )
    Sigma = sample_covariance(X)
    scores, _pairs, _omega = plugin_scores(X, Sigma, edges)
    critical = block_bootstrap_quantile(scores, l, alpha)
    return TestResult(statistic, critical, bool(statistic > critical))
