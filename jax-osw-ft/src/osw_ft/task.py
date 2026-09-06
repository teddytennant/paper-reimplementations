"""A synthetic binary-verifiable task, standing in for the paper's MCQ math set.

Each query is a two-digit addition written as [a, +, b, =] and the expert
trajectory o* is the two-token answer [tens, units]. The verifier is exact
string match, so R(o) in {0, 1} exactly as Assumption 1 requires, and there is
exactly one accepting answer per query.

That last part is what makes p_s(q) computable in closed form: the probability
of sampling a correct trajectory is just pi_theta(o*|q), no enumeration needed.
Tests use that to check behaviour deterministically instead of through noisy
rollout counts.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "PLUS",
    "EQUALS",
    "VOCAB_SIZE",
    "PROMPT_LEN",
    "ANSWER_LEN",
    "SEQ_LEN",
    "build_dataset",
    "exact_match_reward",
]

PLUS = 10
EQUALS = 11
VOCAB_SIZE = 12
PROMPT_LEN = 4
ANSWER_LEN = 2
SEQ_LEN = PROMPT_LEN + ANSWER_LEN


def build_dataset(n_digits=10):
    """All (a, b) pairs with a, b < n_digits, as full [q; o*] token sequences.

    Returns an int array of shape (n_digits**2, SEQ_LEN). Slice [:PROMPT_LEN]
    for the query and [PROMPT_LEN:] for the expert trajectory.
    """
    rows = []
    for a in range(n_digits):
        for b in range(n_digits):
            total = a + b
            rows.append([a, PLUS, b, EQUALS, total // 10, total % 10])
    return np.asarray(rows, dtype=np.int32)


def exact_match_reward(sampled_answer, expert_answer):
    """R(o) in {0, 1}: 1 iff the sampled answer matches o* token for token.

    Works batched; the answer axis is the last one.
    """
    match = jnp.all(jnp.asarray(sampled_answer) == jnp.asarray(expert_answer), axis=-1)
    return match.astype(jnp.result_type(float))
