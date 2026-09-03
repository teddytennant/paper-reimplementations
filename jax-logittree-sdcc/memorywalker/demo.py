"""Run the running example end to end: the tree, the four losses, then SDCC.

    python -m memorywalker.demo

Small enough to finish in seconds on CPU and to serve as an accelerator smoke
test: it does real forward and backward passes and 200 optimizer steps.
"""

from __future__ import annotations

import functools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from .exact import logittree_loss, packed4d_loss  # noqa: E402
from .model import default_config, init_params  # noqa: E402
from .naive import naive_compressed_loss, naive_full_loss  # noqa: E402
from .reference import materialized_tree_loss  # noqa: E402
from .sdcc import leaf_gap, sdcc_loss  # noqa: E402
from .tree import running_example  # noqa: E402


def main() -> None:
    print("device:", jax.devices()[0].platform, jax.devices()[0])
    rollout = running_example()
    cfg = default_config(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    params = init_params(jax.random.key(0), cfg)

    print(f"K={rollout.n_junctions} junctions, "
          f"{len(rollout.response_positions())} response tokens, "
          f"{len(rollout.diverging_leaves())} diverging leaves "
          f"{rollout.diverging_leaves()}")
    for k, branch in enumerate(rollout.branches()):
        print(f"  branch {k}: {branch}  scores {rollout.layer_targets(k)}")

    ref, gref = jax.value_and_grad(materialized_tree_loss)(params, rollout, cfg)
    print(f"\nmaterialized tree   {float(ref):.12f}   (ground truth)")
    for name, fn in [("LogitTree", logittree_loss), ("packed 4D mask", packed4d_loss)]:
        loss, grad = jax.value_and_grad(fn)(params, rollout, cfg)
        worst = max(float(jnp.max(jnp.abs(a - b)))
                    for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(gref)))
        print(f"{name:<19} {float(loss):.12f}   max grad deviation {worst:.2e}")
    for name, fn in [("Naive-Compressed", naive_compressed_loss),
                     ("Naive-Full", naive_full_loss)]:
        loss = float(fn(params, rollout, cfg))
        print(f"{name:<19} {loss:.12f}   bias {loss - float(ref):+.6f}")

    objective = functools.partial(sdcc_loss, rollout=rollout, cfg=cfg, lam=10.0)
    grad_fn = jax.jit(jax.grad(objective))
    report = jax.jit(functools.partial(leaf_gap, rollout=rollout, cfg=cfg))

    def summary(p):
        gaps = [(float(k), float(t)) for k, t in report(p).values()]
        mean_kl = sum(k for k, _ in gaps) / len(gaps)
        max_tv = max(t for _, t in gaps)
        slack = max(t - (max(k, 0.0) / 2) ** 0.5 for k, t in gaps)
        return mean_kl, max_tv, slack

    opt = optax.adam(3e-2)
    state = opt.init(params)
    print("\nSDCC, lambda=10, 200 Adam steps. Pinsker slack is the worst leaf's "
          "TV minus its sqrt(eps/2), so it has to stay negative.")
    kl, tv, slack = summary(params)
    print(f"  step   0: mean KL {kl:.6f}  max TV {tv:.6f}  Pinsker slack {slack:+.6f}")
    for step in range(200):
        updates, state = opt.update(grad_fn(params), state)
        params = optax.apply_updates(params, updates)
        if step % 50 == 49:
            kl, tv, slack = summary(params)
            print(f"  step {step + 1:3d}: mean KL {kl:.6f}  max TV {tv:.6f}  "
                  f"Pinsker slack {slack:+.6f}")


if __name__ == "__main__":
    main()
