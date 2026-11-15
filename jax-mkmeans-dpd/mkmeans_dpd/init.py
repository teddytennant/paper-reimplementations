"""The core-neighbourhood initialization of Section 3.2.1.

The paper is explicit that MK-means DPD is sensitive to the starting centers and
covariances, and gives a seven-step scheme adapted from Melnykov and Melnykov
(2014) that replaces Steps 1 and 2 of Algorithm 2. Points with many close
neighbours are preferred as seeds, a small core is grown around each one, and
the DPD estimator is run on the core to get an initial (mu_k, Sigma_k).

Cluster membership is carried as a boolean mask over all n points rather than as
an index list. That keeps every array shape fixed, so the estimator compiles
once instead of once per cluster size; with index slicing the ten restarts of
the demo spent most of their time in XLA.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .dpd import default_cov_floor, mahalanobis, mdpde, spec_from_cov

# Standard normal 0.99 quantile, used by the Wilson-Hilferty approximation.
_Z99 = 2.3263478740408408


def chi2_quantile_99(p: int) -> float:
    """99th percentile of chi-square with p degrees of freedom.

    Wilson-Hilferty: chi2_p(q) ~= p (1 - 2/(9p) + z_q sqrt(2/(9p)))^3. The worst
    case over p = 1 to 6 is 0.74% high at p = 1 and inside 0.25% elsewhere, and
    it keeps scipy out of the import path. Step 5 of Section 3.2.1 needs only
    this one quantile; the paper writes it as chi2_{p,0.01}, meaning the cutoff
    of the 99% confidence ellipsoid.
    """
    t = 2.0 / (9.0 * p)
    return float(p * (1.0 - t + _Z99 * (t**0.5)) ** 3)


def _mad_cov(pts: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Scaled median absolute deviation, the initial Sigma of Section 3.1.

    Masked-out rows are pushed to the median of the kept ones so they cannot
    drag the two medians around.
    """
    big = jnp.where(mask[:, None], pts, jnp.nan)
    med = jnp.nanmedian(big, axis=0)
    mad = jnp.nanmedian(jnp.abs(big - med), axis=0) * 1.4826
    return jnp.diag(jnp.maximum(mad, 1e-6) ** 2)


@partial(jax.jit, static_argnums=(4, 5, 6))
def _dpd_estimate(x, mask, mu0, sigma0, alpha, cov_floor, n_inner=50):
    """MDPDE of Section 3.1 over the points ``mask`` selects. Step 4."""
    resp = mask.astype(x.dtype)[:, None]
    mu, spec = mdpde(
        x,
        mu0[None, :],
        spec_from_cov(sigma0[None], cov_floor),
        resp,
        alpha,
        cov_floor,
        n_inner,
    )
    return mu[0], spec.sigma[0]


def core_init(x, k, alpha=0.2, key=None, w=20, d_min=20, n_refine=5, cov_floor=None):
    """Steps 1 to 7 of Section 3.2.1. Returns (centers, covariances).

    Step 1 sums the w smallest distances to every point. Step 2 picks a seed
    among the unassigned points with probability proportional to 1/s where s is
    its rank in ascending order of that sum, so dense points are favoured but
    not forced. Steps 3 to 6 alternate "take the D nearest points", "fit the DPD
    estimator", "widen to the 99% confidence ellipsoid". Step 7 removes the
    points used and moves on to the next cluster.

    If the unassigned pool drops below D before K clusters are found, the loop
    stops early; Section 3.2.1 notes this doubles as a signal that the true
    number of clusters is smaller than K. The remaining centers are filled from
    the points furthest from those already chosen, so the caller still gets K.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    if cov_floor is None:
        cov_floor = default_cov_floor(x)
    n, p = x.shape
    w = min(w, n - 1)
    d_min = min(d_min, n)
    if key is None:
        key = jax.random.PRNGKey(0)

    d = jnp.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    s_w = jnp.sum(jnp.sort(d, axis=1)[:, 1 : w + 1], axis=1)  # Step 1

    cutoff = chi2_quantile_99(p)
    inf = jnp.inf
    unassigned = jnp.ones(n, dtype=bool)
    centers, covs = [], []
    keys = jax.random.split(key, k)

    for j in range(k):
        pool = unassigned
        n_pool = int(jnp.sum(pool))
        if n_pool < d_min:
            break

        order = jnp.argsort(jnp.where(pool, s_w, inf))[:n_pool]  # Step 2
        ranks = jnp.arange(1, n_pool + 1, dtype=x.dtype)
        probs = (1.0 / ranks) / jnp.sum(1.0 / ranks)
        mu = x[order[jax.random.choice(keys[j], n_pool, p=probs)]]

        members = _nearest_mask(x, mu, pool, d_min)  # Step 3
        sigma = _mad_cov(x, members)
        for _ in range(n_refine):  # Step 6, repeat Steps 3 to 5
            mu, sigma = _dpd_estimate(x, members, mu, sigma, alpha, cov_floor)
            spec = spec_from_cov(sigma[None], cov_floor)  # Step 5
            inside = pool & (mahalanobis(x, mu[None, :], spec)[:, 0] <= cutoff)
            near = _nearest_mask(x, mu, pool, d_min)  # Step 3, recentred
            # The union with the previous membership is what lets the ellipsoid
            # grow. Re-estimating from the D nearest points alone would reset
            # Sigma to the tight core on every pass and the cluster would never
            # expand past its seed; the paper's phrase is "the data points
            # assigned to the cluster", which by Step 5 is the ellipsoid.
            members = members | inside | near
        mu, sigma = _dpd_estimate(x, members, mu, sigma, alpha, cov_floor)

        centers.append(mu)
        covs.append(sigma)
        unassigned = unassigned & ~members  # Step 7

    while len(centers) < k:  # pool exhausted before K clusters were found
        used = jnp.stack(centers)
        far = jnp.min(jnp.linalg.norm(x[:, None, :] - used[None], axis=-1), axis=1)
        centers.append(x[jnp.argmax(far)])
        covs.append(_mad_cov(x, jnp.ones(n, dtype=bool)))

    return jnp.stack(centers), jnp.stack(covs)


def _nearest_mask(x, mu, pool, d_min):
    """Mask of the ``d_min`` unassigned points closest to ``mu``. Step 3."""
    dd = jnp.where(pool, jnp.linalg.norm(x - mu, axis=-1), jnp.inf)
    return jnp.zeros(x.shape[0], dtype=bool).at[jnp.argsort(dd)[:d_min]].set(True)
