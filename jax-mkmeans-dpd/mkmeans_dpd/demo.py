"""Reproduce the shape of Tables 2 and 3: python -m mkmeans_dpd.demo

Runs K-means, MK-means, MK-means DPD and DC-MK-means DPD on the simulated data
of Section 4.2 and on Iris, and reports accuracy plus the classical and robust
internal indices side by side. DBSCAN and the GMM rows of the paper's tables are
not implemented here.
"""

import argparse

import jax

# Q is a sum of n terms that mostly cancel, so in float32 the last four digits
# of the trace are rounding noise and "did the objective go up" stops being
# answerable. The demo reports that number, so it runs in float64.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .baselines import kmeans, mkmeans
from .cluster import dc_mkmeans_dpd, mkmeans_dpd
from .data import cluster_accuracy, load_iris, make_simulated
from .indices import (
    calinski_harabasz,
    davies_bouldin,
    median_davies_bouldin,
    trimmed_calinski_harabasz,
)

_COLS = ("method", "acc", "DBI", "DBI_M", "CHI", "CHI_trim", "iters", "outliers")


def _n_outliers(x, centers, covs):
    """Count of points outside the 99% ellipsoid of their assigned cluster.

    The Outlier column of Tables 2 and 3: B_i > chi2_{p,0.01} using each
    method's own final center and covariance estimates.
    """
    from .dpd import default_cov_floor, mahalanobis, spec_from_cov
    from .init import chi2_quantile_99

    spec = spec_from_cov(jnp.asarray(covs), default_cov_floor(x))
    b = mahalanobis(jnp.asarray(x), jnp.asarray(centers), spec)
    return int(jnp.sum(jnp.min(b, axis=1) > chi2_quantile_99(x.shape[1])))


def _row(name, x, y, labels, centers, covs, n_iter, trim):
    k = int(jnp.max(labels)) + 1
    return (
        name,
        f"{cluster_accuracy(y, labels, k):.3f}",
        f"{davies_bouldin(x, labels, k):.3f}",
        f"{median_davies_bouldin(x, labels, k):.3f}",
        f"{calinski_harabasz(x, labels, k):.1f}",
        f"{trimmed_calinski_harabasz(x, labels, k, trim):.1f}",
        str(n_iter),
        "-" if covs is None else str(_n_outliers(x, centers, covs)),
    )


def _table(rows):
    widths = [max(len(str(r[i])) for r in (_COLS, *rows)) for i in range(len(_COLS))]
    line = "  ".join(c.rjust(w) for c, w in zip(_COLS, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).rjust(w) for c, w in zip(r, widths)))


def run(x, y, k, alpha, seed, trim, title, n_restarts=10):
    """Fit all four methods from a shared initialization and print the table.

    The remark after Theorem 3.2 says multiple initializations are the standard
    guard against a poor local optimum, so we run the Section 3.2.1 scheme
    ``n_restarts`` times and keep the seed whose DC-MK-means DPD run ends at the
    smallest Q. All four methods then start from that same seed, so the
    comparison is not confounded by a different starting point per method.
    """
    k_key, c_key = jax.random.split(jax.random.PRNGKey(seed))
    dc = dc_mkmeans_dpd(x, k, alpha=alpha, key=c_key, n_init=n_restarts)
    centers, covs0 = dc.init_centers, dc.init_covariances

    km = kmeans(x, k, key=k_key)
    mk = mkmeans(x, k, centers=centers)
    dpd = mkmeans_dpd(x, k, alpha=alpha, centers=centers, covariances=covs0)

    print(f"\n{title}  (n={x.shape[0]}, p={x.shape[1]}, K={k}, alpha={alpha})")
    _table(
        [
            _row("K-means", x, y, km.labels, km.centers, None, km.n_iter, trim),
            _row(
                "MK-means", x, y, mk.labels, mk.centers, mk.covariances, mk.n_iter, trim
            ),
            _row(
                "MK-means DPD",
                x,
                y,
                dpd.labels,
                dpd.centers,
                dpd.covariances,
                dpd.n_iter,
                trim,
            ),
            _row(
                "DC-MK-means DPD",
                x,
                y,
                dc.labels,
                dc.centers,
                dc.covariances,
                dc.n_iter,
                trim,
            ),
        ]
    )
    _monotone("MK-means DPD   ", dpd, "no guarantee, Section 3.3")
    _monotone(
        "DC-MK-means DPD",
        dc,
        f"partition fixed after round {dc.label_fixed_at}, Theorem 3.2",
    )


def _monotone(name, res, note):
    """Count half-steps on which Q went up, above float64 rounding noise."""
    obj = res.objective
    tol = 1e-10 * jnp.abs(obj[:-1])
    ups = int(jnp.sum(jnp.diff(obj) > tol))
    print(f"  {name}: Q rose on {ups} of {obj.size - 1} half-steps ({note})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=0.2, help="DPD tuning parameter")
    ap.add_argument("--trim", type=float, default=0.1, help="CHI_trim trimming level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--restarts", type=int, default=10, help="Section 3.2.1 initializations"
    )
    args = ap.parse_args(argv)

    x, y = make_simulated(seed=args.seed)
    run(
        x,
        y,
        3,
        args.alpha,
        args.seed,
        args.trim,
        "Simulated, Section 4.2",
        args.restarts,
    )

    x, y = load_iris()
    run(x, y, 3, args.alpha, args.seed, args.trim, "Iris, Section 5.1", args.restarts)


if __name__ == "__main__":
    main()
