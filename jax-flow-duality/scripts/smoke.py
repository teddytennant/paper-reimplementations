"""CPU smoke: k_t table plus one tiny DFM train run."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax

from flow_duality.coefficients import k_gaussian, k_negexp, k_uniform
from flow_duality.dfm import train_tiny


def main():
    times = (0.1, 0.3, 0.5, 0.7, 0.9)
    vocabs = (2, 8, 32)
    print("k_t with linear schedule k_tilde = t")
    print(f"{'t':>6} {'family':>10} {'V=2':>10} {'V=8':>10} {'V=32':>10}")
    for t in times:
        row_g = [float(k_gaussian(t, v)) for v in vocabs]
        row_u = [float(k_uniform(t, v, eps=1.0)) for v in vocabs]
        row_e = [float(k_negexp(t, beta=1.0, vocab_size=v)) for v in vocabs]
        print(f"{t:6.1f} {'gaussian':>10} {row_g[0]:10.4f} {row_g[1]:10.4f} {row_g[2]:10.4f}")
        print(f"{t:6.1f} {'uniform':>10} {row_u[0]:10.4f} {row_u[1]:10.4f} {row_u[2]:10.4f}")
        print(f"{t:6.1f} {'negexp':>10} {row_e[0]:10.4f} {row_e[1]:10.4f} {row_e[2]:10.4f}")

    loss0, loss1, _ = train_tiny(jax.random.PRNGKey(0), n_steps=8, batch_size=64)
    print(f"tiny DFM train step: loss0={float(loss0):.4f} loss_after_8={float(loss1):.4f}")


if __name__ == "__main__":
    main()
