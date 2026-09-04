"""MK-means DPD (Algorithm 2) and DC-MK-means DPD (Algorithm 3).

The two differ in exactly one line, the assignment rule of Step 4. Algorithm 2
assigns on the raw squared Mahalanobis distance B_i; Algorithm 3 assigns on the
pointwise DPD loss v_alpha of Equation (3.8), which is the term whose cluster
average is the objective the update step already minimizes. That single change
is what gives Algorithm 3 the joint objective Q of Equation (3.13) and the
finite-step convergence of Theorem 3.2.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .baselines import _euclidean_assign, _ml_covariances, _one_hot, kmeans_plusplus
from .dpd import (
    _PREC,
    ClusterSpec,
    default_cov_floor,
    dpd_gradients,
    dpd_objective,
    joint_objective,
    mahalanobis,
    mdpde,
    pointwise_loss,
    spec_from_cov,
)
from .init import core_init


class DPDResult(NamedTuple):
    """Fitted model plus the traces the convergence theorem is stated about.

    centers:      (K, p)
    covariances:  (K, p, p)
    labels:       (n,)
    objective:    the value of Q (Equation 3.13) after every half-step, in
                  order: initial, assignment, update, assignment, ... For
                  Algorithm 3 this sequence is non-increasing (Proposition 3.1).
                  For Algorithm 2 there is no such guarantee and the trace is
                  reported only so you can watch it fail to be monotone.
    label_history: (n_iter + 1, n) partitions visited, initial first
    n_iter:       number of Step 4 / Step 5 rounds actually run
    label_fixed_at: first round after which the partition never changed again,
                  or None if it was still moving at ``max_iter``. Theorem 3.2
                  says this is finite for Algorithm 3.
    converged:    partition fixed and parameter change below ``tol``
    init_centers, init_covariances: the seed the run actually started from,
                  which with ``n_init > 1`` is the one that won
    """

    centers: jnp.ndarray
    covariances: jnp.ndarray
    labels: jnp.ndarray
    objective: jnp.ndarray
    label_history: jnp.ndarray
    n_iter: int
    label_fixed_at: int | None
    converged: bool
    init_centers: jnp.ndarray
    init_covariances: jnp.ndarray


def _spec_from_precision(lam: jnp.ndarray, cov_floor: float) -> ClusterSpec:
    lam = 0.5 * (lam + jnp.swapaxes(lam, -1, -2))
    evals, evecs = jnp.linalg.eigh(lam)
    evals = jnp.maximum(evals, 1e-12)
    sigma = jnp.einsum("kij,kj,klj->kil", evecs, 1.0 / evals, evecs, precision=_PREC)
    return spec_from_cov(sigma, cov_floor)


def _armijo_candidates(x, mu, spec, resp, alpha, cov_floor, steps):
    """Backtracking steepest-descent candidates in (mu, Lambda), Equation (3.12).

    The paper's Step 5 fallback is a BFGS direction with an Armijo line search.
    We use the steepest-descent direction instead, which is still a descent
    direction, so the accepted step satisfies the same non-increase requirement
    the proof of Proposition 3.1 relies on; only the rate differs. The direction
    is normalized per cluster so one set of step lengths covers both blocks of
    the stacked parameter theta = (mu, vech(Lambda)).
    """
    g_mu, g_lam = dpd_gradients(x, mu, spec, resp, alpha)
    norm = jnp.sqrt(jnp.sum(g_mu**2, axis=-1) + jnp.sum(g_lam**2, axis=(-2, -1)))
    norm = jnp.maximum(norm, 1e-30)
    d_mu = -g_mu / norm[:, None]
    d_lam = -g_lam / norm[:, None, None]

    def one(t):
        m = mu + t * d_mu
        s = _spec_from_precision(spec.lam + t * d_lam, cov_floor)
        return m, s.sigma

    return jax.vmap(one)(steps)


def _safeguarded_update(x, mu, spec, resp, alpha, cov_floor, n_inner, n_backtrack):
    """Step 5 of Algorithm 3, guaranteed not to increase V_k for any cluster.

    Try the warm-started weighted iteration of Section 3.1 first. If it fails to
    return a value of V_{alpha,mu,Sigma}(x_k) at least as small as the starting
    point, walk back along the segment to the previous iterate, then try Armijo
    steepest descent, and finally keep the previous estimate. Every cluster is
    handled independently, which is what the proof does when it multiplies by
    n_k and sums over k.
    """
    p = x.shape[1]
    v_cur = dpd_objective(mahalanobis(x, mu, spec), spec.logdet, resp, alpha, p)

    mu_fp, spec_fp = mdpde(x, mu, spec, resp, alpha, cov_floor, n_inner)
    damp = 0.5 ** jnp.arange(n_backtrack, dtype=x.dtype)

    def blend(t):
        m = mu + t * (mu_fp - mu)
        s = spec.sigma + t * (spec_fp.sigma - spec.sigma)
        return m, s

    mu_d, sig_d = jax.vmap(blend)(damp)
    scale = jnp.sqrt(jnp.mean(jnp.var(x, axis=0)))
    steps = scale * 10.0 ** (-jnp.arange(n_backtrack, dtype=x.dtype))
    mu_g, sig_g = _armijo_candidates(x, mu, spec, resp, alpha, cov_floor, steps)

    cand_mu = jnp.concatenate([mu_d, mu_g, mu[None]], axis=0)
    cand_sig = jnp.concatenate([sig_d, sig_g, spec.sigma[None]], axis=0)

    def value(m, s):
        sp = spec_from_cov(s, cov_floor)
        return dpd_objective(mahalanobis(x, m, sp), sp.logdet, resp, alpha, p)

    vals = jax.vmap(value)(cand_mu, cand_sig)  # (n_cand, K)
    # The last candidate is the incoming estimate, so at least one column entry
    # always passes and the loop cannot get stuck without an admissible step.
    ok = vals <= v_cur[None, :] + 1e-12 * jnp.abs(v_cur)[None, :]
    pick = jnp.argmax(ok, axis=0)
    new_mu = jnp.take_along_axis(cand_mu, pick[None, :, None], axis=0)[0]
    new_sig = jnp.take_along_axis(cand_sig, pick[None, :, None, None], axis=0)[0]
    return new_mu, spec_from_cov(new_sig, cov_floor)


def _assign_dpd(x, mu, spec, prev, alpha):
    """Step 4 of Algorithm 3: arg min_k v_alpha(x_i; mu_k, Sigma_k).

    Ties keep the previous assignment, the fixed consistent rule Theorem 3.2
    asks for; without it a two-cycle between equal-loss clusters could keep Q
    constant forever and the partition would never stabilize.
    """
    v = pointwise_loss(mahalanobis(x, mu, spec), spec.logdet, alpha, x.shape[1])
    best = jnp.argmin(v, axis=1)
    v_prev = jnp.take_along_axis(v, prev[:, None], axis=1)[:, 0]
    v_best = jnp.take_along_axis(v, best[:, None], axis=1)[:, 0]
    return jnp.where(v_prev <= v_best, prev, best)


def _assign_mahalanobis(x, mu, spec, prev, alpha):
    """Step 4 of Algorithm 2: arg min_k B_i, the raw Mahalanobis distance."""
    del alpha
    b = mahalanobis(x, mu, spec)
    best = jnp.argmin(b, axis=1)
    b_prev = jnp.take_along_axis(b, prev[:, None], axis=1)[:, 0]
    b_best = jnp.take_along_axis(b, best[:, None], axis=1)[:, 0]
    return jnp.where(b_prev <= b_best, prev, best)


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def _round(x, mu, sigma, labels, k, alpha, cov_floor, n_inner, n_backtrack, dc):
    """One round of Steps 4 and 5, returning Q after each half-step."""
    p = x.shape[1]
    spec = spec_from_cov(sigma, cov_floor)
    assign = _assign_dpd if dc else _assign_mahalanobis
    new_labels = assign(x, mu, spec, labels, alpha)
    resp = _one_hot(new_labels, k)
    q_assign = joint_objective(mahalanobis(x, mu, spec), spec.logdet, resp, alpha, p)
    if dc:
        # Step 5 of Algorithm 3, with the non-increase safeguard the proof needs.
        new_mu, new_spec = _safeguarded_update(
            x, mu, spec, resp, alpha, cov_floor, n_inner, n_backtrack
        )
    else:
        # Step 5 of Algorithm 2: the plain weighted iteration, no safeguard.
        new_mu, new_spec = mdpde(x, mu, spec, resp, alpha, cov_floor, n_inner)
    q_update = joint_objective(
        mahalanobis(x, new_mu, new_spec), new_spec.logdet, resp, alpha, p
    )
    return new_mu, new_spec.sigma, new_labels, q_assign, q_update


def _fit(
    x,
    k,
    alpha,
    dc,
    key=None,
    centers=None,
    covariances=None,
    n_init=1,
    init="core",
    max_iter=100,
    tol=None,
    cov_floor=None,
    n_inner=25,
    n_backtrack=8,
) -> DPDResult:
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    p = x.shape[1]
    if tol is None:
        # Step 6 compares successive parameter estimates, so the threshold has
        # to sit above the arithmetic's own noise floor. In float32 a parameter
        # of order 1 cannot be pinned down past about 1e-6, and asking for 1e-6
        # makes the loop spin to max_iter long after the partition has settled.
        tol = 1e-9 if x.dtype == jnp.float64 else 1e-5
    if cov_floor is None:
        cov_floor = default_cov_floor(x)
    if centers is None:
        if key is None:
            raise ValueError("pass either `centers` or a PRNG `key`")
        # Step 1, repeated n_init times. The remark after Theorem 3.2 asks for
        # multiple initializations, since the theorem gives a stationary point
        # of Q and not its global minimum. We keep the run that ends lowest.
        best = None
        for sub in jax.random.split(key, n_init):
            if init == "core":
                c0, s0 = core_init(x, k, alpha=alpha, key=sub, cov_floor=cov_floor)
            elif init == "kmeans++":
                c0, s0 = kmeans_plusplus(x, k, sub), None
            else:
                raise ValueError(f"unknown init {init!r}")
            res = _fit(
                x,
                k,
                alpha,
                dc,
                centers=c0,
                covariances=s0,
                max_iter=max_iter,
                tol=tol,
                cov_floor=cov_floor,
                n_inner=n_inner,
                n_backtrack=n_backtrack,
            )
            q = float(res.objective[-1])
            if best is None or q < best[0]:
                best = (q, res)
        return best[1]
    mu = jnp.asarray(centers, dtype=x.dtype)
    init_mu = mu

    labels = _euclidean_assign(x, mu)  # Step 2
    resp = _one_hot(labels, k)
    eye = jnp.broadcast_to(jnp.eye(p, dtype=x.dtype), (k, p, p))
    if covariances is None:
        sigma = _ml_covariances(x, resp, mu, eye)  # Step 3
    else:
        sigma = jnp.asarray(covariances, dtype=x.dtype)
    init_sigma = sigma

    spec0 = spec_from_cov(sigma, cov_floor)
    trace = [
        float(joint_objective(mahalanobis(x, mu, spec0), spec0.logdet, resp, alpha, p))
    ]
    label_hist = [labels]
    fixed_at = None
    converged = False
    n_iter = 0

    for it in range(max_iter):  # Step 6
        prev_mu, prev_sigma, prev_labels = mu, sigma, labels
        mu, sigma, labels, q_a, q_u = _round(
            x, mu, sigma, labels, k, alpha, cov_floor, n_inner, n_backtrack, dc
        )
        trace.extend([float(q_a), float(q_u)])
        label_hist.append(labels)
        n_iter = it + 1

        same = bool(jnp.all(labels == prev_labels))
        fixed_at = fixed_at if same else None
        if same and fixed_at is None:
            fixed_at = it
        delta = max(
            float(jnp.max(jnp.abs(mu - prev_mu))),
            float(jnp.max(jnp.abs(sigma - prev_sigma))),
        )
        # Step 6. Algorithm 3 also waits for the partition to stop
        # moving, which Theorem 3.2 promises will happen; Algorithm 2 has no
        # such promise, so asking for it there can spin until max_iter.
        if delta < tol and (same or not dc):
            converged = True
            break

    return DPDResult(
        centers=mu,
        covariances=spec_from_cov(sigma, cov_floor).sigma,
        labels=labels,
        objective=jnp.asarray(trace),
        label_history=jnp.stack(label_hist),
        n_iter=n_iter,
        label_fixed_at=fixed_at,
        converged=converged,
        init_centers=init_mu,
        init_covariances=init_sigma,
    )


def mkmeans_dpd(x, k, alpha=0.2, **kwargs) -> DPDResult:
    """Algorithm 2: MK-means DPD.

    Steps 1 to 3 seed as in Algorithm 1. Step 4 assigns on the raw Mahalanobis
    distance. Step 5 replaces the MLE with the DPD estimator of Section 3.1.
    Section 3.3 explains why the two steps target different objectives, so the
    reported ``objective`` trace can and does go up as well as down.

    Pass ``centers`` (and optionally ``covariances``) to start from a known
    seed, or a PRNG ``key`` with ``n_init`` and ``init`` in {"core",
    "kmeans++"} to have it seeded for you. ``alpha`` is the DPD tuning
    parameter; the paper uses 0.2 on Iris.
    """
    return _fit(x, k, alpha, dc=False, **kwargs)


def dc_mkmeans_dpd(x, k, alpha=0.2, **kwargs) -> DPDResult:
    """Algorithm 3: Density-Consistent MK-means DPD.

    Identical to Algorithm 2 except that Step 4 assigns on the pointwise DPD
    loss v_alpha of Equation (3.8). Proposition 3.1 gives Q non-increasing at
    every iteration and Theorem 3.2 finite termination of the partition
    sequence, provided the eigenvalues of every Sigma_k stay above some
    delta > 0 (here ``cov_floor``) and ties are broken consistently.
    """
    return _fit(x, k, alpha, dc=True, **kwargs)
