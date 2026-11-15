"""Internal cluster validity indices, classical and the paper's robust versions.

Section 4.1 of arXiv:2608.30093v1. The two new ones are the Median
Davies-Bouldin Index (Equation 4.7) and the Trimmed Calinski-Harabasz Index
(Equation 4.8). Classical DBI (4.3) and CHI (4.4) are here to be the thing the
robust versions are compared against.
"""

import jax.numpy as jnp


def _group_mask(labels: jnp.ndarray, k: int) -> jnp.ndarray:
    return labels[:, None] == jnp.arange(k)[None, :]


def _cluster_means(x, mask):
    nk = jnp.sum(mask, axis=0)
    safe = jnp.where(nk > 0, nk, 1.0)
    return (mask.T.astype(x.dtype) @ x) / safe[:, None], nk


def davies_bouldin(x, labels, k=None) -> float:
    """Classical DBI, Equation (4.3).

        DB = (1/K) sum_i max_{j != i} (sigma_i + sigma_j) / d(c_i, c_j)

    with c_i the cluster centroid and sigma_i the mean Euclidean distance of the
    cluster's points to it. Lower is better.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    k = int(jnp.max(labels)) + 1 if k is None else k
    mask = _group_mask(labels, k)
    centers, nk = _cluster_means(x, mask)
    d = jnp.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
    safe = jnp.where(nk > 0, nk, 1.0)
    sigma = jnp.sum(jnp.where(mask, d, 0.0), axis=0) / safe
    return float(_db_from(centers, sigma, nk))


def median_davies_bouldin(x, labels, k=None) -> float:
    """Median Davies-Bouldin Index DBI_M, Equation (4.7).

    Same shape as (4.3), with the centroid c_i replaced by the coordinate-wise
    median of C_i and the dispersion sigma_i replaced by
    sigma_i^med = median_{x in C_i} d(x, c_i^med). Lower is better.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    k = int(jnp.max(labels)) + 1 if k is None else k
    mask = _group_mask(labels, k)
    nk = jnp.sum(mask, axis=0)

    centers = []
    sigmas = []
    for j in range(k):
        pts = x[labels == j]
        if pts.shape[0] == 0:
            centers.append(jnp.zeros(x.shape[1], dtype=x.dtype))
            sigmas.append(jnp.asarray(0.0, dtype=x.dtype))
            continue
        c = jnp.median(pts, axis=0)
        centers.append(c)
        sigmas.append(jnp.median(jnp.linalg.norm(pts - c, axis=-1)))
    return float(_db_from(jnp.stack(centers), jnp.stack(sigmas), nk))


def _db_from(centers, sigma, nk):
    k = centers.shape[0]
    sep = jnp.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    ratio = (sigma[:, None] + sigma[None, :]) / jnp.where(sep > 0, sep, jnp.inf)
    off = ~jnp.eye(k, dtype=bool)
    live = (nk > 0)[:, None] & (nk > 0)[None, :] & off
    ratio = jnp.where(live, ratio, -jnp.inf)
    worst = jnp.max(ratio, axis=1)
    worst = jnp.where(nk > 0, worst, 0.0)
    return jnp.sum(worst) / jnp.maximum(jnp.sum(nk > 0), 1)


def calinski_harabasz(x, labels, k=None) -> float:
    """Classical CHI, Equation (4.4).

        CH = tr(B_K) / tr(W_K) * (n - K) / (K - 1)

    Higher is better.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    k = int(jnp.max(labels)) + 1 if k is None else k
    mask = _group_mask(labels, k)
    centers, nk = _cluster_means(x, mask)
    grand = jnp.mean(x, axis=0)
    tr_b = jnp.sum(nk * jnp.sum((centers - grand) ** 2, axis=-1))
    resid = x - centers[labels]
    tr_w = jnp.sum(resid * resid)
    n = x.shape[0]
    return float(tr_b / tr_w * (n - k) / (k - 1))


def trimmed_calinski_harabasz(x, labels, k=None, trim=0.1) -> float:
    """Trimmed Calinski-Harabasz Index CHI_trim, Equation (4.8).

    For each cluster, keep C_i^trim = {x in C_i : d(x, c_i) <= q_{1-trim}}, the
    points within the (1 - trim) quantile of the distances to the ordinary
    cluster mean c_i. Then recompute genuine sample means on what is left:
    c_i^trim per cluster and c^trim over all retained points pooled. With

        tr(B^trim) = sum_i n_i^trim ||c_i^trim - c^trim||^2
        tr(W^trim) = sum_i sum_{x in C_i^trim} ||x - c_i^trim||^2

    the identity tr(B^trim) + tr(W^trim) = tr(T^trim) still holds exactly, which
    is the reason the paper trims CHI instead of taking medians the way it does
    for DBI. Higher is better.

    ``trim`` is the paper's alpha in q_{1-alpha}; it is renamed here so it does
    not collide with the DPD tuning parameter. The paper does not report the
    value it used, see the README.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    k = int(jnp.max(labels)) + 1 if k is None else k

    keep_x = []
    keep_lab = []
    for j in range(k):
        pts = x[labels == j]
        if pts.shape[0] == 0:
            continue
        d = jnp.linalg.norm(pts - jnp.mean(pts, axis=0), axis=-1)
        cutoff = jnp.quantile(d, 1.0 - trim)
        sel = d <= cutoff
        keep_x.append(pts[sel])
        keep_lab.append(jnp.full((int(jnp.sum(sel)),), j))
    xt = jnp.concatenate(keep_x)
    lt = jnp.concatenate(keep_lab)

    mask = _group_mask(lt, k)
    centers, nk = _cluster_means(xt, mask)
    grand = jnp.mean(xt, axis=0)
    tr_b = jnp.sum(nk * jnp.sum((centers - grand) ** 2, axis=-1))
    resid = xt - centers[lt]
    tr_w = jnp.sum(resid * resid)
    n_trim = xt.shape[0]
    return float(tr_b / tr_w * (n_trim - k) / (k - 1))
