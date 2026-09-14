"""Generate a synthetic panel and print recovered vs true rho."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from anchor_judge_rho import estimate_from_scores, simulate_panel


def main() -> None:
    key = jax.random.PRNGKey(0)
    true_rho = jnp.array([0.3, 0.7])
    judges, anchors = simulate_panel(
        key,
        n=8000,
        p=4,
        m=2,
        sigma_t2=1.0,
        sigma_c2=0.8,
        rho=true_rho,
        sigma_a=0.9,
    )
    est = estimate_from_scores(judges, anchors)
    print("true rho:     ", jnp.asarray(true_rho))
    print("recovered rho:", est["rho"])
    print("sigma_t2:     ", float(est["sigma_t2"]), "(true 1.0)")
    print("sigma_c2:     ", float(est["sigma_c2"]), "(true 0.8)")
    print("beta:         ", est["beta"])
    print("denominator:  ", est["denominator"])


if __name__ == "__main__":
    main()
