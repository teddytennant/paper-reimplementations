"""Pref-MH and its baselines as scan-based JAX chains.

``pref_mh_chain`` is Algorithm 1 for one judge and Algorithm 2 (Appx. B) for
several. Each step spends a fixed budget of N judge queries per judge, forms
the vote counts K_i, and accepts with the probability in Eq. (8) / Eq. (12).
No pointwise score is ever evaluated and no estimate of a comparison
probability is ever formed.

The two baselines exist for comparison only:

* ``oracle_mh_chain`` is textbook MH on the true log target (Eq. 2). The paper
  calls it Oracle MH; it needs the latent score, which the preference-only
  setting does not give you.
* ``plugin_mh_chain`` is the invalid plug-in sampler of Appx. C.1, which
  estimates the odds as K/(N-K) and drops that into Eq. (7).

The problem is described by three callables plus the judges:

``propose(key, x) -> y``           draw from q(. | x)
``log_q(x_to, x_from)``            log q(x_to | x_from)
``log_p0(x)``                      log base density, up to a constant

so ``log_r0(x, y) = log_p0(y) + log_q(x, y) - log_p0(x) - log_q(y, x)``, the
tractable factor of Eq. (3).
"""

from functools import partial

import jax
import jax.numpy as jnp

from .acceptance import log_alpha_n_vote, log_alpha_plugin
from .judges import count_votes


def log_r0(x, y, log_p0, log_q):
    """r_0(x, y) in logs, the judge-free half of the MH ratio in Eq. (7)."""
    return log_p0(y) + log_q(x, y) - log_p0(x) - log_q(y, x)


def _run(key, x0, step, n_steps):
    """Scan ``step`` for ``n_steps`` and stack the trajectory."""

    def body(carry, k):
        x_next, accepted = step(k, carry)
        return x_next, (x_next, accepted)

    _, (states, accepted) = jax.lax.scan(
        body, x0, jax.random.split(key, n_steps)
    )
    return states, accepted


def _accept(key, log_alpha, x, y):
    """Move to y with probability exp(log_alpha), else stay at x."""
    u = jax.random.uniform(key, dtype=jnp.result_type(float))
    accepted = jnp.log(u) < log_alpha
    x_next = jax.tree.map(lambda a, b: jnp.where(accepted, a, b), y, x)
    return x_next, accepted


@partial(jax.jit, static_argnames=("propose", "log_q", "log_p0", "judges",
                                   "n_votes", "n_steps"))
def pref_mh_chain(key, x0, *, propose, log_q, log_p0, judges, n_votes, n_steps):
    """Algorithm 1 (m = 1) and Algorithm 2 (m > 1).

    ``judges`` is a tuple of ``judge(key, x, y) -> bool``. Each judge is
    queried exactly ``n_votes`` times per proposal, so the per-step comparison
    cost is m * N and does not depend on the states.

    Returns the trajectory x^(1..T) and the accept flags. The initial state is
    not included.
    """

    def step(key, x):
        k_prop, k_judge, k_acc = jax.random.split(key, 3)
        y = propose(k_prop, x)
        keys = jax.random.split(k_judge, len(judges))
        counts = jnp.stack([
            count_votes(k, j, x, y, n_votes) for k, j in zip(keys, judges)
        ])
        la = log_alpha_n_vote(log_r0(x, y, log_p0, log_q), counts, n_votes)
        return _accept(k_acc, la, x, y)

    return _run(key, x0, step, n_steps)


@partial(jax.jit, static_argnames=("propose", "log_q", "log_target", "n_steps"))
def oracle_mh_chain(key, x0, *, propose, log_q, log_target, n_steps):
    """Standard MH with the true unnormalized density, Eq. (2).

    ``log_target`` is log pi up to an additive constant, i.e. log p_0 + s under
    the Gibbs form of Eq. (4). This is the reference chain; it is not available
    when the only access to the target is through pairwise votes.
    """

    def step(key, x):
        k_prop, k_acc = jax.random.split(key, 2)
        y = propose(k_prop, x)
        la = jnp.minimum(
            0.0,
            log_target(y) + log_q(x, y) - log_target(x) - log_q(y, x),
        )
        return _accept(k_acc, la, x, y)

    return _run(key, x0, step, n_steps)


@partial(jax.jit, static_argnames=("propose", "log_q", "log_p0", "judge",
                                   "n_votes", "n_steps"))
def plugin_mh_chain(key, x0, *, propose, log_q, log_p0, judge, n_votes, n_steps):
    """The invalid plug-in sampler of Appx. C.1.

    Same proposal, same judge, same budget as Pref-MH. It estimates the
    win-rate odds by K/(N-K) and plugs that into Eq. (7). Theorem 1 says this
    cannot reproduce the oracle rule; it also breaks detailed balance, so the
    chain converges to something other than pi.
    """

    def step(key, x):
        k_prop, k_judge, k_acc = jax.random.split(key, 3)
        y = propose(k_prop, x)
        k = count_votes(k_judge, judge, x, y, n_votes)
        la = log_alpha_plugin(log_r0(x, y, log_p0, log_q), k, n_votes)
        return _accept(k_acc, la, x, y)

    return _run(key, x0, step, n_steps)
