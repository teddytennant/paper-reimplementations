"""Run Algorithm 1 on the three Appendix B models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax.numpy as jnp

from rash_halpern.halpern import algorithm1
from rash_halpern.models import MODELS


def _fmt(vec) -> str:
    return "[" + ", ".join(f"{float(x):.8g}" for x in vec) + "]"


def main() -> None:
    budgets = (1, 2, 3, 4, 16, 64, 256)
    for model in MODELS:
        label = {"i": "I", "ii": "II", "iii": "III"}[model.kind]
        print(f"Model {label}: {model.name}")
        for n in budgets:
            robust = algorithm1(model, n, nominal=False)
            nominal = algorithm1(model, n, nominal=True)
            err = float(jnp.max(jnp.abs(robust.g_hat - model.g_star)))
            action = model.action_names[int(robust.pi[0])]
            nominal_action = model.action_names[int(nominal.pi[0])]
            print(f"N={n}")
            print(f"  g_hat_N={_fmt(robust.g_hat)}")
            print(f"  max_abs_err={err:.8g}")
            print(f"  action_at_x={action}")
            print(f"  nominal_action_at_x={nominal_action}")
        print()


if __name__ == "__main__":
    main()
