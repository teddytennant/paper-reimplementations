"""Tiny quadratic FCCO smoke: SOX vs MSVRM-v1 vs MSVRM-v2."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from jax_msvr.estimator import correction_gamma
from jax_msvr.metrics import objective
from jax_msvr.msvrm import init_msvrm_state, msvrm_v1_step, msvrm_v2_step
from jax_msvr.problems import make_quadratic
from jax_msvr.sox import sox_step


def _run(step_fn, state, problem, key, T, kwargs):
    for _ in range(T):
        key, sub = jax.random.split(key)
        state = step_fn(state, problem, sub, **kwargs)
    return state


def main():
    m, d, k, B1, T = 8, 16, 2, 2, 200
    problem, _ = make_quadratic(
        m, d, k, jax.random.PRNGKey(0), sigma=0.0, scale=0.25
    )
    w0 = jnp.ones(d)
    F0 = float(objective(problem, w0))
    beta, alpha, eta = 0.3, 0.3, 0.05
    gamma = float(correction_gamma(m, B1, beta))
    common = dict(B1=B1, alpha=alpha, beta=beta, eta=eta)
    results = {}
    for name, step_fn, extra in (
        ("SOX", sox_step, {}),
        ("MSVRM-v1", msvrm_v1_step, {"gamma": gamma}),
        ("MSVRM-v2", msvrm_v2_step, {"gamma": gamma}),
    ):
        state = init_msvrm_state(w0, problem=problem)
        state = _run(step_fn, state, problem, jax.random.PRNGKey(1), T, {**common, **extra})
        FT = float(objective(problem, state.w))
        results[name] = FT
        print(f"{name:10s}  F_0={F0:.6f}  F_T={FT:.6f}")

    finite = all(np.isfinite(v) for v in results.values()) and np.isfinite(F0)
    decreased = results["MSVRM-v1"] < F0
    if not (finite and decreased):
        print("smoke failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
