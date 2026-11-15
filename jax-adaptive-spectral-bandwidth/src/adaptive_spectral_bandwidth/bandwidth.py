"""Per-node Gaussian bandwidth by matching effective rank to dimMST.

Bozkurt and Ortega, "Geometry-Aware Graph Construction via Adaptive
Spectral Bandwidth Control" (arXiv:2609.03306), section 3.

K_SS(sigma) is the pairwise Gaussian kernel on the candidate neighborhood
S, including the center:

    K_ab = exp( - ||x_a - x_b||^2 / (2 sigma^2) )    for a, b in S.

Effective rank is eq (1), Roy and Vetterli:

    r_eff(K) = exp( - sum_j p_j log p_j ),  p_j = lambda_j / sum_k lambda_k.

dimMST is the Costa-Hero GMST scaling estimator used by IDEST: MST length
on n samples from a d-manifold scales as L(n) ~ n^{(d-1)/d}. Fit
log L_i ≈ m log n_i + c on several subsample sizes, then d_hat = 1/(1-m).

Log-log slope ell(sigma) = d log L / d log sigma of the total kernel
energy L(sigma) = sum_{j,k} K_jk(sigma). Criterion eq (2):

    J(sigma) = |r_eff - d_hat| / d_hat  +  (ell_max - ell(sigma)) / ell_max

Grid eq (3): logspace(0.05 * d_{k_mle}, 3.0 * d_{k_cand}), |G|=12.
Weight sharpening eq (4): p = clip((sigma_star / sigma_g)^{-1}, 0.2, 2.0).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


def pairwise_sqdist(x: Array) -> Array:
    """Squared Euclidean distances. Difference-then-square, not a Gram matmul.

    A float32 X X^T on NVIDIA runs at TF32 and can invert a close ranking.
    """
    d = x[:, None, :] - x[None, :, :]
    return jnp.sum(d * d, axis=-1)


def gaussian_kernel(x: Array, sigma: Array | float) -> Array:
    """Pairwise Gaussian kernel K_SS(sigma) on neighborhood points x (n, dim)."""
    sigma = jnp.asarray(sigma)
    dist2 = pairwise_sqdist(x)
    scale = jnp.clip(2.0 * sigma * sigma, 1e-18, None)
    return jnp.exp(-dist2 / scale)


def effective_rank(k: Array) -> Array:
    """Eq (1). Clip tiny negative eigenvalues from symmetry error to 0."""
    evals = jnp.linalg.eigvalsh(0.5 * (k + k.T))
    evals = jnp.clip(evals, 0.0, None)
    total = jnp.sum(evals)
    p = evals / jnp.clip(total, 1e-18, None)
    # 0 log 0 = 0
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))
    return jnp.exp(entropy)


def _mst_length(dist: Array) -> Array:
    """Prim MST total length on a dense distance matrix (n, n).

    n is the candidate pool (paper uses k_cand=30), so O(n^2) is fine.
    """
    n = dist.shape[0]
    inf = jnp.array(jnp.inf, dtype=dist.dtype)
    # Distances to the growing tree. Start from node 0.
    in_tree = jnp.zeros((n,), dtype=bool).at[0].set(True)
    min_d = dist[0].at[0].set(inf)

    def body(i, carry):
        in_tree, min_d, length = carry
        # Pick the closest vertex not yet in the tree.
        masked = jnp.where(in_tree, inf, min_d)
        j = jnp.argmin(masked)
        length = length + masked[j]
        in_tree = in_tree.at[j].set(True)
        min_d = jnp.minimum(min_d, dist[j]).at[j].set(inf)
        min_d = jnp.where(in_tree, inf, min_d)
        return in_tree, min_d, length

    _, _, length = jax.lax.fori_loop(1, n, body, (in_tree, min_d, jnp.array(0.0, dtype=dist.dtype)))
    return length


def mst_length(x: Array) -> Array:
    """Total Euclidean MST length of the point set x."""
    dist = jnp.sqrt(jnp.clip(pairwise_sqdist(x), 0.0, None))
    dist = dist + jnp.eye(x.shape[0]) * 1e12
    return _mst_length(dist)


def dim_mst(x: Array, key: Array, n_sizes: int = 5, n_rep: int = 5) -> Array:
    """Costa-Hero GMST intrinsic-dimension estimator on x (n, dim).

    Subsample sizes are spaced between 8 and n. For each size, n_rep
    random subsets; log L vs log n is fit by least squares, then
    d_hat = 1 / (1 - m) with m the slope (paper: n_rep=5).

    Sizes are Python ints so the MST runs on a static number of points.
    """
    n = int(x.shape[0])
    lo = min(8, n)
    hi = n
    raw = np.round(np.linspace(lo, hi, n_sizes)).astype(int)
    sizes = sorted({int(s) for s in raw if s >= 3})
    if len(sizes) < 2:
        sizes = [max(3, n // 2), n]

    log_n = []
    log_l = []
    k = key
    for size in sizes:
        acc = []
        for _ in range(n_rep):
            k, k_rep = jax.random.split(k)
            perm = jnp.argsort(jax.random.uniform(k_rep, (n,)))
            pts = x[perm[:size]]
            acc.append(mst_length(pts))
        mean_L = jnp.mean(jnp.stack(acc))
        log_n.append(float(np.log(size)))
        log_l.append(jnp.log(jnp.clip(mean_L, 1e-18, None)))
    log_n_a = jnp.array(log_n, dtype=x.dtype)
    log_l_a = jnp.stack(log_l)
    xbar = jnp.mean(log_n_a)
    ybar = jnp.mean(log_l_a)
    m = jnp.sum((log_n_a - xbar) * (log_l_a - ybar)) / jnp.clip(
        jnp.sum((log_n_a - xbar) ** 2), 1e-18, None
    )
    # m = (d-1)/d so d = 1/(1-m). Clip m into (0, 1-eps) for a finite d > 1.
    m = jnp.clip(m, 0.05, 0.95)
    return 1.0 / (1.0 - m)


def loglog_slope(x: Array, sigmas: Array) -> Array:
    """ell(sigma) = d log L / d log sigma, central differences on the grid.

    L(sigma) = sum_{j,k in S} K_jk(sigma).
    """

    def energy(sigma):
        k = gaussian_kernel(x, sigma)
        return jnp.sum(k)

    L = jax.vmap(energy)(sigmas)
    log_L = jnp.log(jnp.clip(L, 1e-18, None))
    log_s = jnp.log(sigmas)
    # Central differences in the interior, one-sided at the ends.
    dlogL = jnp.empty_like(log_L)
    dlogL = dlogL.at[1:-1].set(log_L[2:] - log_L[:-2])
    dlogs = jnp.empty_like(log_s)
    dlogs = dlogs.at[1:-1].set(log_s[2:] - log_s[:-2])
    dlogL = dlogL.at[0].set(log_L[1] - log_L[0])
    dlogs = dlogs.at[0].set(log_s[1] - log_s[0])
    dlogL = dlogL.at[-1].set(log_L[-1] - log_L[-2])
    dlogs = dlogs.at[-1].set(log_s[-1] - log_s[-2])
    return dlogL / jnp.clip(dlogs, 1e-18, None)


def bandwidth_grid(d_mle: Array | float, d_cand: Array | float, n_grid: int = 12) -> Array:
    """Eq (3): logspace(0.05 * d_mle, 3.0 * d_cand)."""
    lo = 0.05 * jnp.asarray(d_mle)
    hi = 3.0 * jnp.asarray(d_cand)
    lo = jnp.clip(lo, 1e-8, None)
    hi = jnp.maximum(hi, lo * 1.01)
    return jnp.logspace(jnp.log10(lo), jnp.log10(hi), n_grid)


def knn_scale(x: Array, center: Array, k: int) -> Array:
    """k-NN Euclidean distance of ``center`` into the neighborhood x."""
    d = jnp.sqrt(jnp.clip(jnp.sum((x - center) ** 2, axis=-1), 0.0, None))
    d_sorted = jnp.sort(d)
    # Skip a possible zero self-distance.
    k = min(k, x.shape[0] - 1)
    return d_sorted[k]


def spectral_criterion(x: Array, sigmas: Array, d_hat: Array | float) -> Array:
    """Eq (2) evaluated on a bandwidth grid."""
    d_hat = jnp.asarray(d_hat)
    d_hat = jnp.clip(d_hat, 1e-6, None)

    def reff_at(sigma):
        return effective_rank(gaussian_kernel(x, sigma))

    reffs = jax.vmap(reff_at)(sigmas)
    ell = loglog_slope(x, sigmas)
    ell_max = jnp.max(ell)
    term1 = jnp.abs(reffs - d_hat) / d_hat
    term2 = (ell_max - ell) / jnp.clip(ell_max, 1e-18, None)
    return term1 + term2


def select_bandwidth(
    x: Array,
    key: Array,
    k_mle: int = 10,
    k_cand: int | None = None,
    n_grid: int = 12,
    d_hat: Array | float | None = None,
) -> tuple[Array, Array, Array]:
    """sigma_star = argmin J(sigma) on the local log-spaced grid.

    x is the neighborhood including the center as row 0.
    Returns (sigma_star, sigmas, J).
    """
    if k_cand is None:
        k_cand = x.shape[0] - 1
    center = x[0]
    d_mle = knn_scale(x, center, k_mle)
    d_c = knn_scale(x, center, k_cand)
    sigmas = bandwidth_grid(d_mle, d_c, n_grid=n_grid)
    if d_hat is None:
        d_hat = dim_mst(x, key)
    J = spectral_criterion(x, sigmas, d_hat)
    star = sigmas[jnp.argmin(J)]
    return star, sigmas, J


def sharpen_weights(weights: Array, sigma_star: Array, sigma_g: Array) -> Array:
    """Eq (4): p = clip((sigma_star / sigma_g)^{-1}, 0.2, 2.0), then w^p / sum w^p."""
    p = jnp.clip(1.0 / jnp.clip(sigma_star / sigma_g, 1e-12, None), 0.2, 2.0)
    w = jnp.clip(weights, 0.0, None)
    wp = jnp.power(w, p)
    return wp / jnp.clip(jnp.sum(wp), 1e-18, None)
