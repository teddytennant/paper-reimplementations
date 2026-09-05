"""The CCE gap the paper reports.

Definition 4 calls the empirical distribution ``Pihat_N = (1/N) sum_n delta_{rho_n}``
an ``eps``-approximate CCE when, for every player,

    sup_{mu^i in Pi_i^gen} (1/N) sum_n [ V^{i,rho_n}(s_1) - V^{i, mu^i . rho_n^{-i}}(s_1) ] <= eps,

and Section 6.1 reports the max of that over players as ``Gap_N``. Two things
make it awkward to compute. The comparator ``mu^i`` is history dependent, and
it is a *single* policy that has to be good on average against all ``N``
opponent profiles at once, so it cannot be chosen per episode.

``general_policy_gap`` computes it exactly. Both obstacles disappear if you
carry the whole vector of episode-indexed reach probabilities down the history
tree. Write ``g_n(tau_h)`` for the probability that history ``tau_h`` is
realized in episode ``n`` with player ``i``'s own action probabilities factored
out. Then

    W(tau_h) = min_{a_i} [ <g(tau_h), l^i_{.,h}(s, a_i)>
                           + sum_{a_{-i}, s'} W(tau_h, (a_i, a_{-i}), s') ],
    g(tau_h, a, s') = g(tau_h) * pi^{-i}_{.,h}(a_{-i}|s) * P_h(s'|s, a),

with ``W = 0`` past layer ``H``. The inner minimization is over the simplex of a
linear function, so a deterministic choice attains it. ``W`` at the root is
``min_{mu^i} (1/N) sum_n V^{i, mu^i . rho_n^{-i}}(s_1)`` exactly, over all
history-dependent deviations. The tree has ``O((|A| |S|)^H)`` nodes, which is
fine at the scale of the paper's benchmark games and hopeless beyond it.

``per_episode_gap`` is the cheaper upper bound that lets the comparator change
every episode. It equals ``(1/N) sum_n [V^{i,rho_n} - min_mu V^{i, mu . rho_n^{-i}}]``,
one backward induction per episode, and it is always at least the exact gap.
"""

from __future__ import annotations

import jax
import numpy as np

from . import values

__all__ = ["general_policy_gap", "per_episode_gap", "cce_gap", "average_values"]


def _batched(game, policy_seq, player):
    ell = jax.vmap(lambda p: values.induced_costs(game, p, player))(policy_seq)
    opp = jax.vmap(lambda p: values.opponent_action_probs(game, p, player))(policy_seq)
    val = jax.vmap(lambda p: values.value(game, p, player))(policy_seq)
    return np.asarray(ell), np.asarray(opp), np.asarray(val)


def average_values(game, policy_seq, player):
    """``(1/N) sum_n V^{i,rho_n}(s_1)``."""
    return float(
        np.mean(np.asarray(jax.vmap(lambda p: values.value(game, p, player))(policy_seq)))
    )


def general_policy_gap(game, policy_seq, player):
    """Exact ``sup_{mu^i in Pi_i^gen}`` gap of Definition 4 for one player."""
    ell, opp, val = _batched(game, policy_seq, player)
    trans = np.asarray(game.transitions)
    n_seq = ell.shape[0]
    n_own = game.n_actions[player]

    flat = np.arange(game.n_joint_actions).reshape(game.n_actions)
    joint_for = [np.take(flat, a, axis=player).reshape(-1) for a in range(n_own)]

    def walk(h, s, g):
        if h == game.horizon:
            return 0.0
        best = np.inf
        for own in range(n_own):
            total = float(g @ ell[:, h, s, own])
            for j in joint_for[own]:
                weight = g * opp[:, h, s, j]
                if weight.max() <= 0.0:
                    continue
                for nxt in np.nonzero(trans[h, s, j] > 0.0)[0]:
                    total += walk(h + 1, int(nxt), weight * trans[h, s, j, nxt])
            best = min(best, total)
        return best

    root = np.full(n_seq, 1.0 / n_seq)
    return float(np.mean(val)) - walk(0, game.s_init, root)


def per_episode_gap(game, policy_seq, player):
    """Upper bound on the gap that lets the deviation change every episode."""
    def one(policies):
        return values.value(game, policies, player) - values.best_response(
            game, policies, player
        )[0]

    return float(np.mean(np.asarray(jax.vmap(one)(policy_seq))))


def cce_gap(game, policy_seq, exact=True):
    """``Gap_N``, the max over players of the per-player gap."""
    fn = general_policy_gap if exact else per_episode_gap
    return max(fn(game, policy_seq, i) for i in range(game.n_players))
