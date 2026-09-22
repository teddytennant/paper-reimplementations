"""Cubic monotone operator, Algorithm 1, unconstrained p=2.

F(x) = x + 0.1 x^3 on R^4, x* = 0, x0 = 0.5 * 1, D = 1, L_2 = M = 1.

T=6 and T=8 leave ||v|| about 0.091 and 0.053 even if every resolvent is
exact, so the printed run uses T=20, where the same Algorithm 1 is below 1e-2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from optimal_mvi.halpern import inexact_accelerated_halpern


def cubic_operator(beta=0.1):
    beta = float(beta)

    def F(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return x + beta * x ** 3

    return F


def run(T=20):
    F = cubic_operator(0.1)
    x0 = 0.5 * jnp.ones(4, dtype=jnp.float64)
    D = float(jnp.linalg.norm(x0))
    L2 = 1.0
    return inexact_accelerated_halpern(F, x0, T=T, L_p=L2, D=D, M=L2, p=2)


def main():
    result = run(T=20)
    print("residual trajectory ||v_t||")
    for t, r in enumerate(result.residuals):
        print(f"t={t} ||v||={float(r):.6e} accepted={result.inner_accepted[t]}")
    print(f"final ||v||={float(result.residuals[-1]):.6e}")
    print(f"all inner steps accepted={result.accepted}")


if __name__ == "__main__":
    main()
