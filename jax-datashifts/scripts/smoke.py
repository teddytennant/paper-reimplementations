"""Tiny synthetic demo (CPU). Not the paper's Novozymes/PACS/ColoredMNIST runs."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax import random

from datashifts import datashifts, plugin_w_beta

jax.config.update("jax_enable_x64", True)


def main() -> None:
    key = random.PRNGKey(0)
    k1, k2, k3, k4 = random.split(key, 4)
    n, d = 64, 4
    x_s = random.normal(k1, (n, d))
    x_t = random.normal(k2, (n, d)) + 2.0
    y_s = jnp.linalg.norm(x_s, axis=1)
    y_t = jnp.linalg.norm(x_t, axis=1) + 1.0
    out = datashifts(x_s, y_s, x_t, y_t, beta=0.01, n_iter=400)
    w_plug = plugin_w_beta(x_s, x_t, beta=0.01, n_iter=400)
    print(f"plugin W_beta = {float(w_plug):.4f}")
    print(f"S_Cov (debiased) = {float(out.s_cov):.4f}")
    print(f"S_Cpt = {float(out.s_cpt):.4f}")


if __name__ == "__main__":
    main()
