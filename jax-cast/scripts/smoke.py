"""Build CAST-CHOL on a small graph Laplacian and print a PCG residual."""

from __future__ import annotations

import sys

import jax
import numpy as np

from jax_cast.chol import (
    cast_chol,
    conductance_laplacian,
    factor_preconditioner,
    pcg,
)


def cycle_conductance(n: int, w: float = 1.0) -> np.ndarray:
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        c[i, j] = w
        c[j, i] = w
    return c


def main() -> int:
    n = 8
    rho = 2
    w = cycle_conductance(n)
    a = conductance_laplacian(w)
    factor = cast_chol(w, rho=rho, key=jax.random.PRNGKey(0))
    b = np.linspace(-1.0, 1.0, n)
    b = b - b.mean()
    precon = factor_preconditioner(factor)
    _x, r, rel = pcg(a, b, precon, maxiter=3, tol=0.0)
    _xu, _ru, rel_u = pcg(a, b, lambda v: v, maxiter=3, tol=0.0)
    print(f"n={n} rho={rho} columns={len(factor.stars)}")
    print(f"pcg_rel_residual={rel[-1]:.6e} iters={len(rel) - 1}")
    print(f"cg_rel_residual={rel_u[-1]:.6e} iters={len(rel_u) - 1}")
    print(f"pcg_abs_residual={float(np.linalg.norm(r)):.6e}")
    if not (rel[-1] < rel[0] and rel[-1] < rel_u[-1]):
        print("PCG residual did not drop below CG", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
