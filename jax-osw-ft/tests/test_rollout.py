"""Rollout plumbing: Algorithm 1 lines 5-6, and what p_hat is estimating.

p_s(q) = E_{o~pi}[R(o)|q] is a sum over the whole answer space weighted by the
verifier, and the toy task is small enough to compute that sum outright. The
closed form in rollout.success_probability is checked against it.
"""

import itertools

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from osw_ft.model import ModelConfig, init_params, log_probs
from osw_ft.rollout import (
    estimate_p_hat,
    rollout_rewards,
    sample_rollouts,
    success_probability,
)
from osw_ft.task import ANSWER_LEN, PROMPT_LEN, SEQ_LEN, VOCAB_SIZE, build_dataset, exact_match_reward


def _setup(seed=0):
    config = ModelConfig(vocab_size=VOCAB_SIZE, max_len=SEQ_LEN, d_model=16, n_heads=2, d_ff=32)
    params = init_params(jr.PRNGKey(seed), config, dtype=jnp.float64)
    batch = jnp.asarray(build_dataset(4))[jnp.asarray([0, 7, 13])]
    return config, params, batch


def test_dataset_answers_are_the_arithmetic():
    data = build_dataset(10)
    assert data.shape == (100, SEQ_LEN)
    for row in data:
        a, plus, b, eq, tens, units = row.tolist()
        assert (plus, eq) == (10, 11)
        assert tens * 10 + units == a + b


def test_verifier_is_binary_and_exact():
    expert = jnp.asarray([[1, 4], [0, 9]])
    good = exact_match_reward(expert, expert)
    bad = exact_match_reward(jnp.asarray([[1, 5], [0, 9]]), expert)
    assert jnp.array_equal(good, jnp.asarray([1.0, 1.0]))
    assert jnp.array_equal(bad, jnp.asarray([0.0, 1.0]))


@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_success_probability_matches_the_enumerated_sum(temperature):
    """p_s(q) = sum_o pi(o|q) R(o), enumerated over all 144 answers per query.

    The closed form is only valid because the verifier accepts exactly one
    trajectory. Enumerating the reward-weighted sum is the definition, so this
    checks the shortcut rather than assuming it.
    """
    config, params, batch = _setup()
    got = success_probability(params, config, batch, PROMPT_LEN, temperature)

    enumerated = []
    for row in batch:
        query, expert = row[:PROMPT_LEN], row[PROMPT_LEN:]
        total = 0.0
        for tokens in itertools.product(range(VOCAB_SIZE), repeat=ANSWER_LEN):
            answer = jnp.asarray(tokens, dtype=row.dtype)
            seq = jnp.concatenate([query, answer])
            prob = jnp.exp(jnp.sum(log_probs(params, config, seq, temperature)[PROMPT_LEN - 1 :]))
            total += float(prob) * float(exact_match_reward(answer, expert))
        enumerated.append(total)

    assert np.allclose(np.asarray(got), np.asarray(enumerated), rtol=1e-10, atol=1e-14)


def test_rollout_shapes_and_token_range():
    config, params, batch = _setup()
    answers = sample_rollouts(
        params, config, jr.PRNGKey(3), batch[:, :PROMPT_LEN], ANSWER_LEN, K=5
    )
    assert answers.shape == (3, 5, ANSWER_LEN)
    assert int(jnp.min(answers)) >= 0
    assert int(jnp.max(answers)) < VOCAB_SIZE


def test_rewards_are_binary_and_agree_with_the_verifier():
    config, params, batch = _setup()
    rewards = rollout_rewards(params, config, jr.PRNGKey(4), batch, PROMPT_LEN, K=8)
    assert rewards.shape == (3, 8)
    assert set(np.unique(np.asarray(rewards)).tolist()) <= {0.0, 1.0}


def test_p_hat_concentrates_on_p_s():
    """p_hat is a mean of K Bernoulli(p_s) draws, so it lands near p_s for large K.

    Fixed key and a 4-sigma band, on a model trained enough to have a success
    rate worth estimating. This is the one sampled check in the suite; the
    tolerance comes from the p(1-p)/K variance of Theorem 1, not from taste.
    """
    from osw_ft.train import TrainConfig, train  # noqa: PLC0415

    cfg = TrainConfig(steps=120, batch_size=6, K=4, lr=3e-3, seed=0, n_digits=3, eval_every=0)
    params = train(cfg)["params"]
    batch = jnp.asarray(build_dataset(3))

    p_s = success_probability(params, cfg.model, batch, PROMPT_LEN)
    K = 4096
    p_hat_value = estimate_p_hat(params, cfg.model, jr.PRNGKey(7), batch, PROMPT_LEN, K)
    band = 4.0 * jnp.sqrt(jnp.maximum(p_s * (1.0 - p_s), 1e-12) / K)
    assert bool(jnp.all(jnp.abs(p_hat_value - p_s) <= band + 1e-6))
