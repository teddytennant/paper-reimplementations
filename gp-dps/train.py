"""CPU smoke: linear+sine signal in D=6, DPS with small B, write results.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from gp_dps._host import preload

preload()

import jax
import jax.numpy as jnp
import numpy as np

from gp_dps.select import dps


def _data(n: int = 40, d: int = 6, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    kx, ky = jax.random.split(key)
    X = jax.random.uniform(kx, (n, d), minval=0.0, maxval=1.0)
    y = (
        2.2 * X[:, 0]
        + 1.6 * jnp.sin(2.0 * jnp.pi * X[:, 1])
        + 0.12 * jax.random.normal(ky, (n,))
    )
    return X, y


def main() -> None:
    key = jax.random.PRNGKey(0)
    kd, kr = jax.random.split(key)
    X, y = _data(key=kd)
    out = dps(X, y, kr, B=32, q=0.2, procedure="by", steps=40, lr=0.08)
    selected = [int(i) for i in np.asarray(out["selected"])]
    nu = [float(v) for v in np.asarray(out["nu"])]
    p = [float(v) for v in np.asarray(out["p"])]
    ell = [float(v) for v in np.asarray(out["ell"])]
    payload = {
        "selected": selected,
        "nu": nu,
        "p": p,
        "ell": ell,
        "eta": float(out["eta"]),
        "sigma2": float(out["sigma2"]),
        "B": 32,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
    }
    path = Path(__file__).resolve().parent / "results.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print("selected:", selected)
    print("nu:", [round(v, 4) for v in nu])
    print("p:", [round(v, 4) for v in p])
    print("wrote", path)


if __name__ == "__main__":
    main()
