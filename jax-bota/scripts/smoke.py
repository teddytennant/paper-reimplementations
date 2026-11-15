"""CPU smoke: 2-source/2-sink topology costs and a tiny 2D GMM BFM loop."""

from __future__ import annotations

from jax_bota.cost import two_source_two_sink_costs
from jax_bota.train import sample_gmm_pair, train_bota
from jax import random


def main() -> None:
    for alpha in (1.0, 0.5):
        independent, branched = two_source_two_sink_costs(alpha)
        print(
            f"alpha={alpha} independent={float(independent):.6f} "
            f"branched={float(branched):.6f}"
        )

    x0, x1 = sample_gmm_pair(random.PRNGKey(0), n=16, dim=2)
    result = train_bota(
        x0,
        x1,
        alpha=0.5,
        n_time=4,
        n_discrete_opt=12,
        n_bfm_steps=80,
        hidden=24,
        lr_bfm=8e-3,
        seed=0,
    )
    init = float(result.losses[0])
    final = float(result.losses[-1])
    print(f"L_BFM init={init:.6f} final={final:.6f}")
    print(f"discrete loss={float(result.discrete.loss):.6f}")


if __name__ == "__main__":
    main()
