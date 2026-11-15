"""Layered optimistic online mirror descent, Algorithms 1 and 2.

Both algorithms are radically uncoupled: player ``i``'s state is its own
``(x_tilde, Q)`` per ``(layer, state)``, and the only thing it ever reads is its
own feedback. Nothing here passes another player's policy, value or random bits
into an update.

Algorithm 1 (full feedback), one episode ``t``:

    x_{t,h}^{i,s}       = P_{x_tilde_{t-1}}(Q^i_{t-1,h}(s, .))     (optimistic step)
    pi^i_{t,h}(. | s)   = x_{t,h}^{i,s}                            (played policy)
    x_tilde_{t,h}^{i,s} = P_{x_tilde_{t-1}}(Q^i_{t,h}(s, .))       (mirror update)

The optimism is that the played iterate takes a step on the *previous* episode's
Q vector, the best available prediction of the current one, while the secondary
sequence ``x_tilde`` takes the same step on the realized Q.

Algorithm 2 (partial feedback) changes exactly two things: the mirror steps run
over the floored simplex ``Delta_i^zeta``, and the exact ``Q`` is replaced by
the blocked trajectory estimate of Section 3.2. Within a block the profile is
frozen for ``B`` independent episodes, and

    Qhat^i_{k,h}(s, a_i) = (1/N_{k,i}(s,a_i)) sum_r 1{s_h^{k,r}=s, a_h^{i,k,r}=a_i} Y_h^{i,k,r}

with ``Y^{i,k,r}_h = sum_{tau >= h} l^i_tau(s_tau, a_tau)`` the realized tail
cost, and ``0`` when the pair was never visited.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import values
from .mirror import prox_step, step_sizes

__all__ = [
    "smoothing",
    "run_full_feedback",
    "run_partial_feedback",
    "sample_block",
    "estimate_q",
]


def smoothing(game):
    """``lambda_i = 1/|A_i|``, the choice made in line 2 of both algorithms."""
    return tuple(1.0 / n for n in game.n_actions)


def _init_state(game, init):
    """``x_tilde_0 = argmin Psi_i``, the uniform policy, and ``Q_0 = 0``.

    ``init`` overrides the starting iterate. Line 3 of both algorithms starts at
    uniform; a symmetric game with a symmetric equilibrium then has uniform as a
    fixed point of the whole dynamic, so tests that want to watch the iterates
    travel need to start somewhere else.
    """
    xt = values.uniform_policies(game) if init is None else tuple(init)
    q0 = tuple(jnp.zeros((game.horizon, game.n_states, n)) for n in game.n_actions)
    return xt, q0


def _prox_all(grad, base, eta, lam, floor):
    """Apply the mirror step at every ``(layer, state)`` with the layer's step size."""
    return prox_step(grad, base, eta[:, None], lam, floor)


def run_full_feedback(game, n_episodes, eta0=0.5, lam=None, init=None):
    """Run Algorithm 1 for ``T`` episodes.

    Returns the played profiles as a tuple with one ``(T, H, S, |A_i|)`` array
    per player. The dynamics are deterministic: full feedback hands every player
    the exact ``Q^{i,pi_t}``, so no randomness enters.
    """
    eta = step_sizes(game.horizon, n_episodes, eta0)
    lam = smoothing(game) if lam is None else lam
    state = _init_state(game, init)

    def body(carry, _):
        xt, q_prev = carry
        played = tuple(
            _prox_all(q_prev[i], xt[i], eta, lam[i], 0.0)
            for i in range(game.n_players)
        )
        q_now = tuple(
            values.player_values(game, played, i)[0] for i in range(game.n_players)
        )
        xt_new = tuple(
            _prox_all(q_now[i], xt[i], eta, lam[i], 0.0)
            for i in range(game.n_players)
        )
        return (xt_new, q_now), played

    _, traj = jax.lax.scan(body, state, None, length=n_episodes)
    return traj


def sample_block(game, policies, n_episodes, key):
    """Roll out ``B`` independent episodes under a frozen product profile.

    Returns ``(states, own_actions, tail_costs)`` with shapes ``(H, B)``,
    ``(m, H, B)`` and ``(m, H, B)``; ``tail_costs[i, h, r]`` is ``Y^{i,r}_h``.
    """
    m, hor = game.n_players, game.horizon
    strides = np.concatenate([np.cumprod(game.n_actions[::-1])[::-1][1:], [1]])
    s = jnp.full((n_episodes,), game.s_init, dtype=jnp.int32)
    states, actions, costs = [], [], []
    for h in range(hor):
        key, ks = jax.random.split(key)
        keys = jax.random.split(key, m)
        acts = [
            jax.random.categorical(keys[i], jnp.log(policies[i][h][s]), axis=-1)
            for i in range(m)
        ]
        joint = sum(int(strides[i]) * acts[i] for i in range(m))
        states.append(s)
        actions.append(jnp.stack(acts))
        costs.append(game.costs[h][:, s, joint])
        s = jax.random.categorical(ks, jnp.log(game.transitions[h][s, joint]), axis=-1)
    costs = jnp.stack(costs, axis=1)  # (m, H, B)
    tail = jnp.cumsum(costs[:, ::-1], axis=1)[:, ::-1]
    return jnp.stack(states), jnp.stack(actions, axis=1), tail


def estimate_q(game, rollout):
    """Form ``Qhat^i_{k,h}`` from a block of trajectories."""
    states, actions, tail = rollout
    hor, n_episodes = states.shape
    layer = jnp.repeat(jnp.arange(hor), n_episodes)
    flat_s = states.reshape(-1)
    out = []
    for i in range(game.n_players):
        shape = (hor, game.n_states, game.n_actions[i])
        idx = (layer, flat_s, actions[i].reshape(-1))
        counts = jnp.zeros(shape).at[idx].add(1.0)
        sums = jnp.zeros(shape).at[idx].add(tail[i].reshape(-1))
        out.append(jnp.where(counts > 0, sums / jnp.maximum(counts, 1.0), 0.0))
    return tuple(out)


def run_partial_feedback(
    game, n_blocks, block_length, key, eta0=0.1, zeta=0.05, lam=None, init=None
):
    """Run Algorithm 2 for ``K`` blocks of ``B`` episodes each.

    Returns the per-block profiles, one ``(K, H, S, |A_i|)`` array per player.
    Episode ``(k-1)B + r`` played profile ``pi_k``, so an average over blocks is
    also an average over the ``N = KB`` episodes.
    """
    if not 0.0 < zeta < 1.0:
        raise ValueError("zeta must lie in (0, 1); zeta = 1 freezes the policy")
    eta = step_sizes(game.horizon, n_blocks, eta0)
    lam = smoothing(game) if lam is None else lam
    floors = tuple(zeta / n for n in game.n_actions)
    state = _init_state(game, init)

    def body(carry, k):
        xt, q_prev, rng = carry
        rng, sub = jax.random.split(rng)
        played = tuple(
            _prox_all(q_prev[i], xt[i], eta, lam[i], floors[i])
            for i in range(game.n_players)
        )
        rollout = sample_block(game, played, block_length, sub)
        q_now = estimate_q(game, rollout)
        xt_new = tuple(
            _prox_all(q_now[i], xt[i], eta, lam[i], floors[i])
            for i in range(game.n_players)
        )
        return (xt_new, q_now, rng), played

    _, traj = jax.lax.scan(body, state + (key,), jnp.arange(n_blocks))
    return traj
