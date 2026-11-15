"""Section 2.1: the optimal control variate and the score identity it rests on.

The derivation of the weight has two moving parts. Subtracting a baseline leaves
the policy gradient unbiased because E_{o~pi}[grad log pi(o|q)] = 0, and the
baseline that minimizes the MSE of the scalar signal is c* = E[R|q] = p_s, which
for a binary reward is also the oracle advantage A(o*, q) = 1 - p_s of eq (7).

The score identity is checked by summing over the entire trajectory space of a
deliberately tiny model, not by sampling.
"""

import itertools
from fractions import Fraction

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from osw_ft.model import ModelConfig, init_params, log_probs
from osw_ft.weighting import optimal_control_variate_mse

P_GRID = [Fraction(i, 8) for i in range(9)]
C_GRID = [Fraction(i, 8) for i in range(9)]


@pytest.mark.parametrize("c", C_GRID)
@pytest.mark.parametrize("p", P_GRID)
def test_control_variate_mse_decomposes_around_p(p, c):
    """eq (4): E[(R - c)^2] = (c - p)^2 + p(1-p), exactly.

    Written this way the minimizer is visible: the only c-dependent term is
    (c - p)^2, so c* = p = p_s(q) and the residual is Var[R].
    """
    mse = p * (1 - c) ** 2 + (1 - p) * c**2
    assert mse == (c - p) ** 2 + p * (1 - p)
    assert float(optimal_control_variate_mse(float(p), float(c))) == pytest.approx(
        float(mse), rel=1e-14, abs=1e-15
    )


@pytest.mark.parametrize("p", P_GRID)
def test_c_star_is_the_strict_minimizer(p):
    """c* = p_s beats every other c on the grid, strictly, unless c = c*."""
    best = p * (1 - p) ** 2 + (1 - p) * p**2
    for c in C_GRID:
        mse = p * (1 - c) ** 2 + (1 - p) * c**2
        if c == p:
            assert mse == best
        else:
            assert mse > best


@pytest.mark.parametrize("p", P_GRID)
def test_c_star_equals_the_oracle_advantage_weight(p):
    """eq (7): with R(o*) = 1, A(o*, q) = 1 - c* = 1 - p_s, which is the weight."""
    c_star = p
    assert 1 - c_star == 1 - p


def _enumerate_trajectories(params, config, query, answer_len, temperature=1.0):
    """Every possible answer, its probability, and its score, for a tiny model."""
    prompt_len = query.shape[0]

    def logp(params, answer):
        seq = jnp.concatenate([query, answer])
        return jnp.sum(log_probs(params, config, seq, temperature)[prompt_len - 1 :])

    grad_fn = jax.grad(logp)
    for tokens in itertools.product(range(config.vocab_size), repeat=answer_len):
        answer = jnp.asarray(tokens, dtype=query.dtype)
        yield jnp.exp(logp(params, answer)), grad_fn(params, answer)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_score_function_expectation_is_zero(seed):
    """E_{o~pi}[grad_theta log pi(o|q)] = 0, by summing the whole trajectory space.

    This is the identity section 2.1 needs for subtracting c(q) to leave the
    gradient unbiased. Vocabulary 4 and answer length 2 gives 16 trajectories,
    so the expectation is a finite sum with no sampling error in it at all.
    """
    config = ModelConfig(vocab_size=4, max_len=3, d_model=8, n_heads=2, d_ff=16)
    params = init_params(jr.PRNGKey(seed), config, dtype=jnp.float64)
    query = jnp.asarray([2], dtype=jnp.int32)

    total_prob = 0.0
    accumulated = jax.tree.map(jnp.zeros_like, params)
    for prob, score in _enumerate_trajectories(params, config, query, answer_len=2):
        total_prob += float(prob)
        accumulated = jax.tree.map(lambda a, s, p=prob: a + p * s, accumulated, score)

    assert total_prob == pytest.approx(1.0, abs=1e-12)
    worst = max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree.leaves(accumulated))
    assert worst < 1e-12


def test_score_expectation_is_not_zero_under_a_different_distribution():
    """Control: the identity is about pi's own expectation, not any weighting.

    Reweighting the same scores by a uniform distribution over trajectories
    leaves a nonzero sum, so the test above is measuring something.
    """
    config = ModelConfig(vocab_size=4, max_len=3, d_model=8, n_heads=2, d_ff=16)
    params = init_params(jr.PRNGKey(0), config, dtype=jnp.float64)
    query = jnp.asarray([2], dtype=jnp.int32)

    accumulated = jax.tree.map(jnp.zeros_like, params)
    n = 0
    for _prob, score in _enumerate_trajectories(params, config, query, answer_len=2):
        accumulated = jax.tree.map(lambda a, s: a + s, accumulated, score)
        n += 1
    worst = max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree.leaves(accumulated))
    assert n == 16
    assert worst > 1e-6
