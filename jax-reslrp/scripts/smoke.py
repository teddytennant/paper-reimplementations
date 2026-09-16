"""CPU smoke: cancellation case plus 2-block residual-MLP attribution."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from reslrp.explain import attribute
from reslrp.measures import amplification, cancellation, relevance_amplification
from reslrp.models import init_residual_mlp, residual_mlp
from reslrp.residual import lrp0, lrp_gamma


def main() -> None:
    z_in = jnp.asarray(1.0)
    z_up = jnp.asarray(-1.0 + 1e-4)
    z_out = z_in + z_up
    r_out = jnp.asarray(1.0)
    c = cancellation(z_in, z_up)
    a_fwd = amplification(z_in, z_out)
    print("near-cancellation C", float(c), "A_fwd", float(a_fwd))

    r0_in, r0_up = lrp0(z_in, z_up, r_out, eps=0.0)
    print("LRP-0 max|R|", float(jnp.maximum(jnp.abs(r0_in), jnp.abs(r0_up))))
    for gamma in (0.5, 1.0):
        r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma=gamma)
        a = relevance_amplification(r_in, r_up, r_out)
        max_r = float(jnp.maximum(jnp.abs(r_in), jnp.abs(r_up)))
        print(f"LRP-γ={gamma} max|R|", max_r, "A", float(a))

    key = random.PRNGKey(0)
    params = init_residual_mlp(key, d_in=16, d_hidden=32, n_classes=4, n_blocks=2)
    x = random.normal(random.PRNGKey(1), (4, 16))
    class_idx = 0
    logits = residual_mlp(params, x)
    r = attribute(params, x, class_idx, gamma=1.0)
    cons = jnp.sum(r, axis=-1) - logits[:, class_idx]
    print("conservation error", float(jnp.max(jnp.abs(cons))))
    print("input relevance energy", float(jnp.sum(r**2)))


if __name__ == "__main__":
    main()
