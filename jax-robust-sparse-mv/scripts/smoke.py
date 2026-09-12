"""Smoke run of (P1) on a tiny synthetic instance (CPU)."""

from __future__ import annotations

import ctypes
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_LIBSTDCXX = "/run/current-system/sw/share/nix-ld/lib/libstdc++.so.6"
if os.path.isfile(_LIBSTDCXX):
    ctypes.CDLL(_LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)

from jax import config

config.update("jax_enable_x64", True)

from robust_sparse_mv.bnb import branch_and_bound_p1
from robust_sparse_mv.brute import brute_force_p1
from robust_sparse_mv.closed_form import assumption2_holds, robust_constraint_value
from robust_sparse_mv.problem import synthetic_instance


def main() -> None:
    problem = synthetic_instance(n=6, seed=0, beta=0.04)
    print(f"N={problem.n}  gamma={problem.gamma}  r_bar={problem.r_bar}  beta={problem.beta}")
    print(f"Assumption 2: {bool(assumption2_holds(problem))}")
    brute = brute_force_p1(problem)
    bnb = branch_and_bound_p1(problem)
    lhs = float(robust_constraint_value(problem, bnb.x))
    print(f"brute  F_beta={brute.f_beta:.8g}  support={brute.support}  xi_evals={brute.n_xi_evals}")
    print(
        f"bnb    F_beta={bnb.f_beta:.8g}  support={bnb.support}  "
        f"xi_evals={bnb.n_xi_evals}  pruned_right={bnb.n_right_pruned}"
    )
    print(f"robust constraint lhs={lhs:.8g}  (target {problem.r_bar})")


if __name__ == "__main__":
    main()
