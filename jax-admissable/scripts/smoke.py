#!/usr/bin/env python
"""Short training loop (hundreds of env steps, not 1e6)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax_admissable.train import smoke_train


def main():
    p_miss = smoke_train()
    print("p_miss", p_miss.tolist())
    print("critical_index", 0, "p_miss[0]", float(p_miss[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
