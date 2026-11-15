"""Toy CPU/GPU smoke: frozen random-walk sampler plus a distance target."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from qrg.kernel import LOCAL, PREDNEXT, qrg_step
from qrg.quotients import pairwise_distance


def _distance(x):
    q, _ = pairwise_distance(x, 0, 1)
    return q


def _rollout(x0, keys, sigma, target, mode):
    x = x0
    quotient = lambda z: pairwise_distance(z, 0, 1)
    for key in keys:
        v = sigma * jax.random.normal(key, x.shape)
        if mode is None:
            x = x + v
        else:
            x = qrg_step(x, v, quotient, target, rho=1.0, mode=mode)
    return x


def main():
    n_atoms = 4
    n_steps = 48
    sigma = 0.08
    target = 1.5
    key = jax.random.PRNGKey(0)
    key, k_x = jax.random.split(key)
    x0 = jax.random.normal(k_x, (n_atoms, 3))
    keys = jax.random.split(key, n_steps)

    x_unguided = _rollout(x0, keys, sigma, target, mode=None)
    x_local = _rollout(x0, keys, sigma, target, mode=LOCAL)
    x_pred = _rollout(x0, keys, sigma, target, mode=PREDNEXT)

    d0 = float(_distance(x0))
    results = {
        "n_atoms": n_atoms,
        "n_steps": n_steps,
        "sigma": sigma,
        "target_distance": target,
        "initial_distance": d0,
        "unguided_distance": float(_distance(x_unguided)),
        "local_qrg_distance": float(_distance(x_local)),
        "prednext_qrg_distance": float(_distance(x_pred)),
        "local_abs_error": abs(float(_distance(x_local)) - target),
        "unguided_abs_error": abs(float(_distance(x_unguided)) - target),
        "device": str(jax.devices()[0]),
    }
    out = Path("results.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
