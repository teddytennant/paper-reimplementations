"""Density power divergence pieces: the objective, its gradients, and the
estimating equations that the update step of every DPD algorithm here solves.

Equation numbers refer to Mondal, Banerjee and Mandal, "Robust K-means
Clustering using the Density Power Divergence Measure", arXiv:2608.30093v1.

Notation follows the paper: alpha is the DPD tuning parameter, p the data
dimension, B_i the squared Mahalanobis distance (x_i - mu)' Sigma^-1 (x_i - mu),
omega_i = exp(-alpha/2 B_i) the DPD weight, Lambda = Sigma^-1 the precision.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Every matmul that feeds a Mahalanobis distance is pinned to HIGHEST.
# On an H200 a float32 matmul defaults to TF32, which keeps only 10 explicit
# mantissa bits. Squared distances between nearby cluster centers differ in the
# 3rd or 4th significant digit, so TF32 error can be larger than the gap and
# flip which cluster wins the argmin. HIGHEST forces true float32 accumulation
# (and is a no-op on CPU, where the default is already exact).
_PREC = jax.lax.Precision.HIGHEST


class ClusterSpec(NamedTuple):
    """Per-cluster covariance in the three forms the algorithms need.

    sigma:   (K, p, p) covariance with eigenvalues floored at ``cov_floor``
    lam:     (K, p, p) precision Lambda = Sigma^-1
    whiten:  (K, p, p) W with W W' = Lambda, so B = ||W' (x - mu)||^2
    logdet:  (K,) log |Sigma|
    """

    sigma: jnp.ndarray
    lam: jnp.ndarray
    whiten: jnp.ndarray
    logdet: jnp.ndarray


def default_cov_floor(x, rel: float = 1e-3) -> float:
    """A data-scaled value for delta, the eigenvalue floor of Theorem 3.2.

    The theorem needs *some* delta > 0, and the reason is not cosmetic: without
    it V_alpha is unbounded below, since C_alpha(Sigma) grows like |Sigma|^{-a/2}
    and a cluster that collapses onto p + 1 points drives Q to minus infinity.
    An absolute constant cannot serve, because "small" depends on the units, so
    the default is ``rel`` times the mean eigenvalue of the global sample
    covariance, tr(S) / p. On Iris that is about 1e-3, well under the smallest
    within-species eigenvalue, so it never binds on a real cluster.
    """
    return float(rel * jnp.mean(jnp.var(jnp.asarray(x), axis=0)))


def spec_from_cov(sigma: jnp.ndarray, cov_floor: float) -> ClusterSpec:
    """Symmetrize and floor the eigenvalues of each covariance.

    The floor is the constant ``delta`` of Theorem 3.2: the finite-convergence
    proof assumes every Sigma_k produced by the algorithm has eigenvalues
    bounded below by some delta > 0, which is what keeps C_alpha(Sigma) bounded
    (Equation 3.17) and the objective bounded below. Enforcing it here rather
    than hoping for it makes the theorem's hypothesis true by construction.
    """
    sigma = 0.5 * (sigma + jnp.swapaxes(sigma, -1, -2))
    evals, evecs = jnp.linalg.eigh(sigma)
    evals = jnp.maximum(evals, cov_floor)
    sigma = jnp.einsum("kij,kj,klj->kil", evecs, evals, evecs, precision=_PREC)
    whiten = evecs * (evals**-0.5)[:, None, :]
    lam = jnp.einsum("kij,klj->kil", whiten, whiten, precision=_PREC)
    return ClusterSpec(sigma, lam, whiten, jnp.sum(jnp.log(evals), axis=-1))


def mahalanobis(x: jnp.ndarray, mu: jnp.ndarray, spec: ClusterSpec) -> jnp.ndarray:
    """B[i, k] = (x_i - mu_k)' Sigma_k^-1 (x_i - mu_k), Step 4 of Algorithm 1.

    Returns an (n, K) array. Computed as a squared norm of the whitened
    difference so the result is a sum of squares and cannot go negative.
    """
    dx = x[:, None, :] - mu[None, :, :]
    z = jnp.einsum("nkj,kjd->nkd", dx, spec.whiten, precision=_PREC)
    return jnp.sum(z * z, axis=-1)


def _a_const(alpha: float, p: int) -> jnp.ndarray:
    """A = (1 + alpha)^{(p+2)/2} / alpha, the constant in Equations (3.8), (3.10)."""
    return jnp.asarray((1.0 + alpha) ** ((p + 2) / 2.0) / alpha)


def _c_const(alpha: float, p: int, logdet: jnp.ndarray) -> jnp.ndarray:
    """C_alpha(Sigma) = (2 pi)^{-p alpha/2} |Sigma|^{-alpha/2} (1+alpha)^{-p/2}.

    Defined just above Equation (3.7); this is the determinant-carrying factor
    that makes dV/dSigma^-1 a genuine equation in Sigma^-1 rather than one the
    precision drops out of, which is the whole reason DC-MK-means converges.
    """
    log_c = (
        -0.5 * p * alpha * jnp.log(2.0 * jnp.pi)
        - 0.5 * alpha * logdet
        - 0.5 * p * jnp.log1p(alpha)
    )
    return jnp.exp(log_c)


def pointwise_loss(
    b: jnp.ndarray, logdet: jnp.ndarray, alpha: float, p: int
) -> jnp.ndarray:
    """Pointwise DPD loss, Equation (3.8).

        v_alpha(x; mu, Sigma) = C_alpha(Sigma) [1 - A exp(-alpha/2 B)]

    ``b`` is (n, K), ``logdet`` is (K,); the result is (n, K). This is the
    quantity DC-MK-means DPD assigns on (Step 4 of Algorithm 3) and whose
    average over a cluster is exactly V_alpha (Equation 3.7).
    """
    c = _c_const(alpha, p, logdet)
    a = _a_const(alpha, p)
    return c[None, :] * (1.0 - a * jnp.exp(-0.5 * alpha * b))


def dpd_objective(
    b: jnp.ndarray,
    logdet: jnp.ndarray,
    resp: jnp.ndarray,
    alpha: float,
    p: int,
) -> jnp.ndarray:
    """Per-cluster DPD objective V_{alpha,mu,Sigma}(x_k), Equation (3.5).

        V = C_alpha(Sigma) [1 - (A / m) sum_i omega_i]

    ``resp`` is the (n, K) 0/1 membership matrix, so m = n_k is the cluster
    size. Empty clusters get 0. Returns a (K,) array.
    """
    c = _c_const(alpha, p, logdet)
    a = _a_const(alpha, p)
    omega = jnp.exp(-0.5 * alpha * b)
    nk = jnp.sum(resp, axis=0)
    swo = jnp.sum(resp * omega, axis=0)
    safe_nk = jnp.where(nk > 0, nk, 1.0)
    return jnp.where(nk > 0, c * (1.0 - a * swo / safe_nk), 0.0)


def joint_objective(
    b: jnp.ndarray,
    logdet: jnp.ndarray,
    resp: jnp.ndarray,
    alpha: float,
    p: int,
) -> jnp.ndarray:
    """Q(C, {mu_k, Sigma_k}) of Equation (3.13).

        Q = sum_k sum_{i in C_k} v_alpha(x_i; mu_k, Sigma_k) = sum_k n_k V_k

    Theorem 3.2 says this is non-increasing at every iteration of Algorithm 3
    and that the partition reaching a fixed point takes finitely many steps.
    """
    return jnp.sum(resp * pointwise_loss(b, logdet, alpha, p))


def dpd_gradients(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    spec: ClusterSpec,
    resp: jnp.ndarray,
    alpha: float,
):
    """Gradients of V_{alpha,mu,Sigma}(x_k), Equations (3.10) and (3.11).

        grad_mu V = -(K A alpha / m) |Lambda|^{alpha/2} Lambda sum_i omega_i (x_i - mu)
        grad_Lambda V = (K alpha / 2) |Lambda|^{alpha/2}
                        [Sigma (1 - (A/m) sum_i omega_i)
                         + (A/m) sum_i omega_i (x_i - mu)(x_i - mu)']

    with K = (2 pi)^{-p alpha/2} (1 + alpha)^{-p/2}. Setting both to zero gives
    back the estimating equations (3.6) and (3.9), which is the check that the
    weighted update really is a stationary point of the objective.

    Returns (grad_mu, grad_lambda) of shapes (K, p) and (K, p, p). Both are
    per-cluster gradients of V_k, not of Q.
    """
    p = x.shape[1]
    a = _a_const(alpha, p)
    k_const = (2.0 * jnp.pi) ** (-0.5 * p * alpha) * (1.0 + alpha) ** (-0.5 * p)
    det_pow = jnp.exp(-0.5 * alpha * spec.logdet)  # |Lambda|^{alpha/2}

    b = mahalanobis(x, mu, spec)
    omega = jnp.exp(-0.5 * alpha * b) * resp
    nk = jnp.sum(resp, axis=0)
    m = jnp.where(nk > 0, nk, 1.0)

    dx = x[:, None, :] - mu[None, :, :]
    s1 = jnp.einsum("nk,nkj->kj", omega, dx, precision=_PREC)  # sum omega (x - mu)
    s2 = jnp.einsum("nk,nki,nkj->kij", omega, dx, dx, precision=_PREC)
    swo = jnp.sum(omega, axis=0)

    scale = k_const * det_pow
    g_mu = -(scale * a * alpha / m)[:, None] * jnp.einsum(
        "kij,kj->ki", spec.lam, s1, precision=_PREC
    )
    g_lam = (0.5 * scale * alpha)[:, None, None] * (
        spec.sigma * (1.0 - a * swo / m)[:, None, None] + (a / m)[:, None, None] * s2
    )
    return g_mu, g_lam


def weighted_step(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    spec: ClusterSpec,
    resp: jnp.ndarray,
    alpha: float,
    cov_floor: float,
    denom_floor: float = 1e-3,
):
    """One sweep of the iterative weighted MDPDE equations (3.6) / (3.9).

        omega_i = exp(-alpha/2 B_i)
        mu      = sum_i omega_i x_i / sum_i omega_i
        Sigma^-1 = (sum_i omega_i - m alpha / (1+alpha)^{(p+2)/2})
                   (sum_i omega_i (x_i - mu)(x_i - mu)')^-1

    Note m alpha / (1+alpha)^{(p+2)/2} = m / A. At alpha = 0 the weights are all
    1 and that correction vanishes, so this collapses to the sample mean and the
    MLE covariance, the update used in Step 5 of Algorithm 1.

    Clusters with no points keep their incoming parameters. The denominator is
    clipped at ``denom_floor * n_k``: if every point in a cluster is far enough
    out that its weight underflows, the correction term can drive it to zero or
    below and the covariance would come back indefinite.
    """
    p = x.shape[1]
    a = _a_const(alpha, p)

    b = mahalanobis(x, mu, spec)
    u = resp * jnp.exp(-0.5 * alpha * b)
    su = jnp.sum(u, axis=0)
    nk = jnp.sum(resp, axis=0)
    nonempty = (nk > 0) & (su > 0)

    safe_su = jnp.where(su > 0, su, 1.0)
    mu_new = jnp.einsum("nk,nj->kj", u, x, precision=_PREC) / safe_su[:, None]
    mu_new = jnp.where(nonempty[:, None], mu_new, mu)

    dx = x[:, None, :] - mu_new[None, :, :]
    scatter = jnp.einsum("nk,nki,nkj->kij", u, dx, dx, precision=_PREC)
    denom = jnp.maximum(su - nk / a, denom_floor * jnp.maximum(nk, 1.0))
    sigma_new = scatter / denom[:, None, None]
    sigma_new = jnp.where(nonempty[:, None, None], sigma_new, spec.sigma)
    return mu_new, spec_from_cov(sigma_new, cov_floor)


def mdpde(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    spec: ClusterSpec,
    resp: jnp.ndarray,
    alpha: float,
    cov_floor: float,
    n_inner: int = 25,
):
    """Run the weighted equations (3.6) to a fixed point, warm-started at ``mu``.

    Section 3.1: "the MDPDE can alternatively be obtained by solving the
    following equations iteratively". Algorithm 3 Step 5 requires the warm
    start, since the monotonicity argument of Proposition 3.1 compares the
    result against the value at the previous iterate.
    """

    def body(_, carry):
        m, s = carry
        return weighted_step(x, m, s, resp, alpha, cov_floor)

    return jax.lax.fori_loop(0, n_inner, body, (mu, spec))
