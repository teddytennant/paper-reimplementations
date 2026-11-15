import jax
import jax.numpy as jnp
import numpy as np

from tauleap_schedule.sampler import (
    MASK,
    product_bernoulli_predictor,
    run_tau_leap,
    tau_leap_alg1,
    tau_leap_alg2,
)
from tauleap_schedule.schedules import geometric, q_from_beta, uniform


def test_alg1_alg2_block_size_and_final_reveal():
    N = 10
    q_k = 0.35
    n_draws = 4000
    predictor = product_bernoulli_predictor(0.5)

    def count_alg2(key):
        x = jnp.full((N,), MASK, dtype=jnp.int32)
        mask = jnp.ones((N,), dtype=bool)
        _, mask2 = tau_leap_alg2(x, mask, q_k, key, predictor)
        return (mask.sum() - mask2.sum()).astype(jnp.float32)

    keys = jax.random.split(jax.random.PRNGKey(0), n_draws)
    counts = jax.vmap(count_alg2)(keys)
    mean_s = float(np.asarray(counts).mean())
    expected = N * q_k
    assert abs(mean_s - expected) < 0.05

    def count_alg1(key):
        x = jnp.full((N,), MASK, dtype=jnp.int32)
        mask = jnp.ones((N,), dtype=bool)
        _, mask2 = tau_leap_alg1(x, mask, q_k, key, predictor)
        return (mask.sum() - mask2.sum()).astype(jnp.float32)

    counts1 = jax.vmap(count_alg1)(jax.random.split(jax.random.PRNGKey(1), n_draws))
    mean_s1 = float(np.asarray(counts1).mean())
    assert abs(mean_s1 - expected) < 0.05

    beta = uniform(6)
    q = q_from_beta(beta)
    q_geo = q_from_beta(geometric(6, a=0.5, N=N))
    np.testing.assert_allclose(q, q_from_beta(beta))
    assert q.shape == q_geo.shape
    assert abs(q[-1] - 1.0) < 1e-12

    x0 = jnp.full((N,), MASK, dtype=jnp.int32)
    mask0 = jnp.ones((N,), dtype=bool)
    x1, m1 = run_tau_leap(
        tau_leap_alg1, x0, mask0, q, jax.random.PRNGKey(2), predictor
    )
    x2, m2 = run_tau_leap(
        tau_leap_alg2, x0, mask0, q, jax.random.PRNGKey(3), predictor
    )
    assert not bool(np.asarray(m1).any())
    assert not bool(np.asarray(m2).any())
    assert int(np.asarray((x1 == MASK).sum())) == 0
    assert int(np.asarray((x2 == MASK).sum())) == 0
