"""Smoke: Dodis pairs, Alg 1 on a monotone allocation, Alg 2/3 on DSIC prices."""

from __future__ import annotations

import numpy as np

from ic_tester import (
    dodis_pairs,
    myerson_price_1d,
    nu2,
    coordinatewise_monotonicity,
    dsic,
    dsic_1d,
)


def main() -> None:
    print("nu2(8) =", nu2(8))
    print("dodis_pairs(3) =", dodis_pairs(3))

    rng = np.random.default_rng(0)

    def x_mono(bids):
        bids = np.asarray(bids)
        return np.minimum(1, bids)

    r1 = coordinatewise_monotonicity(x_mono, n=2, m=4, eps=0.5, rng=rng)
    print("alg1 monotone accepted=", r1.accepted, "queries=", r1.n_queries)

    x1 = np.array([0, 0, 1, 1, 1])
    p1 = myerson_price_1d(x1)

    def a1(b):
        return int(x1[b]), float(p1[b])

    r2 = dsic_1d(a1, m=4, d=1, eps=0.5, rng=rng)
    print("alg2 dsic accepted=", r2.accepted, "queries=", r2.n_queries)

    thresh = 2

    def a_n(bids):
        bids = np.asarray(bids, dtype=int)
        x = (bids >= thresh).astype(int)
        p = np.where(x, float(thresh - 1), 0.0)
        return x, p

    r3 = dsic(a_n, n=2, m=4, d=1, eps=0.5, rng=rng)
    print("alg3 dsic accepted=", r3.accepted, "queries=", r3.n_queries)


if __name__ == "__main__":
    main()
