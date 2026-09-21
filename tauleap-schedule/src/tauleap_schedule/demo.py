"""python -m tauleap_schedule.demo"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from tauleap_schedule.eps_fact import eps_fact
from tauleap_schedule.functionals import D, DTC, TC
from tauleap_schedule.schedules import (
    geometric,
    limit_smooth,
    q_from_beta,
    shoot_finite_k,
    stationary_k2,
    uniform,
)


def _rho(u):
    u = np.asarray(u, dtype=np.float64)
    return 2.0 + u


def main() -> None:
    N = 16
    K = 8
    print("rho(u) = 2 + u, N =", N, "K =", K)
    print("D   =", D(_rho))
    print("TC  =", TC(_rho, N))
    print("DTC =", DTC(_rho, N))

    x = stationary_k2(_rho)
    print("K=2 stationary x =", x)

    uni = uniform(K)
    shot = shoot_finite_k(_rho, K)
    geo = geometric(K, a=0.5, N=N)
    t = np.linspace(0.0, 1.0, K + 1)
    sm = limit_smooth(_rho, t)

    print("eps_fact uniform   =", eps_fact(uni, _rho, N))
    print("eps_fact shot      =", eps_fact(shot, _rho, N))
    print("eps_fact geometric =", eps_fact(geo, _rho, N))
    print("eps_fact smooth    =", eps_fact(sm, _rho, N))
    print("q_from_beta(geo)   =", q_from_beta(geo))
    print("shot beta          =", shot)


if __name__ == "__main__":
    main()
