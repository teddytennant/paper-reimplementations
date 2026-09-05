"""Value functions, induced single-player MDPs, and best responses.

Everything here is the Section 2.2 bookkeeping that Algorithm 1 consumes:

    Q^{i,pi}_h(s, a_i) = E_{a_{-i} ~ pi^{-i}_h(.|s)}[ l^i_h(s, a)
                          + sum_{s'} P_h(s'|s, a) V^{i,pi}_{h+1}(s') ]
    V^{i,pi}_h(s)      = E_{a_i ~ pi^i_h(.|s)}[ Q^{i,pi}_h(s, a_i) ],  V_{H+1} = 0.

Fixing the opponents at ``pi^{-i}`` turns player ``i``'s problem into the
induced MDP ``M^i`` with cost ``l^i_{t,h}(s,a_i)`` and kernel
``P^i_{t,h}(s'|s,a_i)``, both obtained by averaging over ``a_{-i}``. Its
optimal value is player ``i``'s best response value, which is what the CCE gap
compares against.

A policy profile is a tuple with one array per player, player ``i``'s having
shape ``(H, S, |A_i|)``. Every function here is a pure function of arrays, so
``jax.vmap`` over a leading iterate axis works.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "uniform_policies",
    "joint_action_probs",
    "opponent_action_probs",
    "induced_costs",
    "induced_transitions",
    "player_values",
    "value",
    "best_response",
]


def uniform_policies(game):
    """The profile in which every player randomizes uniformly at every state."""
    return tuple(
        jnp.full((game.horizon, game.n_states, n), 1.0 / n) for n in game.n_actions
    )


def _outer(policies, n_actions, skip=None):
    """Product over players of ``pi^j_h(a_j|s)``, laid out as a flat joint index.

    ``skip=i`` replaces player ``i``'s factor by ones, which gives
    ``prod_{j != i} pi^j_h(a_j|s)`` at every joint action.
    """
    out = None
    for j, pol in enumerate(policies):
        factor = jnp.ones_like(pol) if j == skip else pol
        out = factor if out is None else out[..., :, None] * factor[..., None, :]
        out = out.reshape(out.shape[:2] + (-1,))
    return out


def joint_action_probs(game, policies):
    """``prod_j pi^j_h(a_j|s)`` with shape ``(H, S, A)``."""
    return _outer(policies, game.n_actions)


def opponent_action_probs(game, policies, player):
    """``prod_{j != i} pi^j_h(a_j|s)`` with shape ``(H, S, A)``."""
    return _outer(policies, game.n_actions, skip=player)


def _marginalize(x, n_actions, player, action_axis):
    """Sum a joint-action axis down to player ``player``'s own action axis."""
    shape = x.shape[:action_axis] + tuple(n_actions) + x.shape[action_axis + 1:]
    x = x.reshape(shape)
    axes = tuple(
        action_axis + j for j in range(len(n_actions)) if j != player
    )
    return x.sum(axis=axes)


def induced_costs(game, policies, player):
    """``l^i_{t,h}(s, a_i)``, shape ``(H, S, |A_i|)``."""
    opp = opponent_action_probs(game, policies, player)
    weighted = game.costs[:, player] * opp
    return _marginalize(weighted, game.n_actions, player, 2)


def induced_transitions(game, policies, player):
    """``P^i_{t,h}(s' | s, a_i)``, shape ``(H, S, |A_i|, S)``."""
    opp = opponent_action_probs(game, policies, player)
    weighted = game.transitions * opp[..., None]
    return _marginalize(weighted, game.n_actions, player, 2)


def player_values(game, policies, player):
    """Return ``(Q, V)`` for one player: shapes ``(H, S, |A_i|)`` and ``(H+1, S)``.

    This is the exact ``Q^{i,pi}_h`` that Algorithm 1 receives under full
    feedback (Definition 1).
    """
    ell = induced_costs(game, policies, player)
    trans = induced_transitions(game, policies, player)
    pol = policies[player]

    def body(v_next, h):
        q = ell[h] + jnp.einsum("sax,x->sa", trans[h], v_next)
        v = jnp.einsum("sa,sa->s", pol[h], q)
        return v, (q, v)

    v_final = jnp.zeros(game.n_states)
    _, (qs, vs) = jax.lax.scan(
        body, v_final, jnp.arange(game.horizon), reverse=True
    )
    return qs, jnp.concatenate([vs, v_final[None]], axis=0)


def value(game, policies, player):
    """``V^{i,pi}(s_1)``, a scalar."""
    return player_values(game, policies, player)[1][0, game.s_init]


def best_response(game, policies, player):
    """Optimal value and a greedy Markov policy in player ``i``'s induced MDP.

    Returns ``(V*(s_1), policy)`` where ``policy`` has shape ``(H, S, |A_i|)``.
    Because the induced MDP is Markov, no history-dependent deviation can beat
    this value against a *single fixed* opponent profile.
    """
    ell = induced_costs(game, policies, player)
    trans = induced_transitions(game, policies, player)

    def body(v_next, h):
        q = ell[h] + jnp.einsum("sax,x->sa", trans[h], v_next)
        v = q.min(axis=-1)
        greedy = jax.nn.one_hot(q.argmin(axis=-1), q.shape[-1], dtype=q.dtype)
        return v, (v, greedy)

    v_final = jnp.zeros(game.n_states)
    _, (vs, greedy) = jax.lax.scan(
        body, v_final, jnp.arange(game.horizon), reverse=True
    )
    return vs[0, game.s_init], greedy
