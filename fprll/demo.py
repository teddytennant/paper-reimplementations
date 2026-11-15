"""Short switching-comparator run of the density-mixture master."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from fprll.simulate import run


def main():
    T = 32
    seg = 16
    u = np.array([1.0 if (t // seg) % 2 == 0 else -1.0 for t in range(T)])
    g = -u
    out = run(g, gamma=1.0, sigma=1.0, lam=0.25, comparators=u, seed=0)
    print("T=32 gamma=1 sigma=1 lambda=0.25 seed=0")
    for H, reg, switches in zip(out["scales"], out["scale_regret"], out["scale_switches"]):
        print(f"H={H} indicator-augmented regret {reg:.3f} switches {int(switches)}")
    print(
        f"mixture indicator-augmented regret {out['mixture_regret']:.3f} "
        f"switches {int(out['mixture_switches'])}"
    )


if __name__ == "__main__":
    main()
