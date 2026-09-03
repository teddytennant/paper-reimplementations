"""Stochastic pairwise judges.

A judge is a callable ``judge(key, x, y) -> bool`` returning one binary vote:
True means the judge prefers the proposal y to the current state x. That is
the only channel the sampler has to the target. It never sees a score, a
density, or a comparison probability.

``bradley_terry_judge`` builds one from a latent score function so the sampler
can be checked against a target we know in closed form. Under Eq. (5) its
votes are Bernoulli(sigma(s(y) - s(x))), which is exactly the assumption
Theorem 2 needs, and the induced target is the Gibbs form of Eq. (4),
pi(x) ~ p_0(x) e^{s(x)}.
"""

import jax
import jax.numpy as jnp

from .acceptance import bt_win_prob


def bradley_terry_judge(score_fn):
    """A BT judge with latent score ``score_fn``, following Eq. (5).

    The returned callable draws one Bernoulli vote. The score itself stays
    inside the closure; callers of the judge only ever see a bit.
    """

    def judge(key, x, y):
        p = bt_win_prob(score_fn(x), score_fn(y))
        return jax.random.bernoulli(key, p)

    return judge


def count_votes(key, judge, x, y, n_votes):
    """Query ``judge`` independently N times on (x, y) and return K.

    Algorithm 1, lines 4 and 5: K = sum of the N votes favouring y. The budget
    is fixed and spent in full, there is no early stop and no rejection loop.
    """
    keys = jax.random.split(key, n_votes)
    votes = jax.vmap(judge, in_axes=(0, None, None))(keys, x, y)
    return jnp.sum(votes.astype(jnp.int32))
