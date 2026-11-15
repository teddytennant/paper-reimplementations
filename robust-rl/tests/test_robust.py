import jax
import jax.numpy as jnp

from robust_rl import (
    bellman_residual,
    conformal_quantile,
    contamination_next,
    init_params,
    maze_kernel,
    nominal_backup,
    robust_backup,
    stepwise_score,
    warm_start_q,
)


def test_kernel_is_a_distribution_and_reward_in_unit_interval():
    P, r = maze_kernel(goal=24, slip=0.1)
    assert P.shape == (25, 4, 25)
    assert r.shape == (25, 4)
    assert jnp.allclose(P.sum(axis=-1), 1.0)
    assert jnp.all(r >= 0.0) and jnp.all(r <= 1.0)


def test_robust_backup_is_more_pessimistic_than_nominal():
    Q = jnp.zeros((25, 4)).at[0, 0].set(1.0)
    P, r = maze_kernel(goal=24)
    robust = robust_backup(Q, r, P, 0.9, 0.4)
    nominal = nominal_backup(Q, r, P, 0.9)
    assert jnp.all(robust <= nominal + 1e-5)
    assert jnp.any(robust < nominal - 1e-4)


def test_one_sample_contamination_matches_the_paper():
    V = jnp.arange(25, dtype=jnp.float32)
    observed = V[3]
    got = contamination_next(V, observed, 0.4)
    expect = 0.4 * jnp.min(V) + 0.6 * observed
    assert jnp.allclose(got, expect)
    assert got < observed


def test_conformal_quantile_covers_the_claimed_rank():
    scores = jnp.array([0.1, 0.2, 0.4, 0.5, 1.0, 1.2, 1.5, 2.0, 3.0])
    q = conformal_quantile(scores, 0.1)
    # N=9, rank = ceil(0.9*10) = 9, the largest finite score
    assert jnp.isfinite(q)
    assert float(q) == 3.0
    assert float(q) >= float(jnp.max(scores)) - 1e-6


def test_conformal_quantile_uses_infinity_when_rank_exceeds_n():
    scores = jnp.array([0.2, 0.4, 0.6, 0.8])
    q = conformal_quantile(scores, 0.1)
    # N=4, rank = ceil(0.9*5) = 5 > 4
    assert not jnp.isfinite(q)


def test_bellman_residual_calls_the_same_operator():
    Q = jax.random.normal(jax.random.PRNGKey(0), (25, 4))
    P, r = maze_kernel(goal=12)
    residual = bellman_residual(Q, r, P, 0.9, 0.2)
    expected = jnp.max(jnp.abs(Q - robust_backup(Q, r, P, 0.9, 0.2)))
    assert jnp.allclose(residual, expected)


def test_warm_start_q_is_finite_and_shaped():
    params = init_params(jax.random.PRNGKey(1))
    Q = warm_start_q(24, 0.2, params)
    assert Q.shape == (25, 4)
    assert jnp.all(jnp.isfinite(Q))


def test_stepwise_score_matches_the_formula():
    got = stepwise_score(1.5, 2.0, 0.2, 3, 0.9)
    allowance = 2.0 * (0.8 ** 3) / 0.1
    assert jnp.allclose(got, max(0.0, 1.5 - allowance))
