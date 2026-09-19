"""CPU smoke: synthetic 2D stumps, a few Algorithm 1 steps, print certificates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_dis_smv.bounds import dis_r_certificate, dis_v_certificate
from jax_dis_smv.majority_vote import empirical_risk_01
from jax_dis_smv.train import algorithm_1, axis_aligned_stumps, synthetic_binary_2d


def main():
    key = jax.random.PRNGKey(0)
    k_data, k_train = jax.random.split(key)
    X, y = synthetic_binary_2d(k_data, n=40)
    voter_preds = axis_aligned_stumps(X, thresholds_per_dim=3)
    m = int(voter_preds.shape[1])
    beta = jnp.full((m,), 0.5)
    result = algorithm_1(
        k_train,
        voter_preds,
        y,
        beta,
        delta=0.05,
        n_epochs=8,
        eta=0.15,
        objective="dis-r",
        slope=100.0,
    )
    n = int(y.shape[0])
    rhat = empirical_risk_01(voter_preds, result.rho, y)
    dis_r = dis_r_certificate(rhat, result.alpha, beta, result.rho, n, 0.05)
    dis_v = dis_v_certificate(rhat, result.alpha, beta, n, 0.05, 1.5)
    print(f"voters={m} n={n}")
    print(f"train 0-1={float(rhat):.4f}")
    print(f"DIS-R={float(dis_r):.4f}")
    print(f"DIS-V={float(dis_v):.4f}")
    print(f"min alpha={float(jnp.min(result.alpha)):.4f}")


if __name__ == "__main__":
    main()
