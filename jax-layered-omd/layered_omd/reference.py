"""Slow reference computations used to check the fast ones.

Nothing in here is on the algorithm's path. These are independent routes to the
same numbers: brute-force enumeration of every trajectory of a layered game,
and the minimax value of a two-player constant-sum layered game from backward
induction with a matrix-game linear program at each state.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "enumerate_trajectories",
    "expected_costs",
    "minimax_value",
    "joint_action_index",
    "all_joint_actions",
    "product_probs",
]


def enumerate_trajectories(game, policies):
    """Yield ``(prob, states, joint_actions)`` for every trajectory with ``prob > 0``.

    ``states`` has length ``H + 1`` and ``joint_actions`` length ``H``; the
    actions are flat joint indices. Only usable on the toy games here, since the
    count is ``O((|S| |A|)^H)``.
    """
    probs = np.asarray(product_probs(policies))
    trans = np.asarray(game.transitions)

    def walk(h, states, actions, p):
        if h == game.horizon:
            yield p, tuple(states), tuple(actions)
            return
        s = states[-1]
        for j in range(game.n_joint_actions):
            pa = probs[h, s, j]
            if pa <= 0.0:
                continue
            for nxt in np.nonzero(trans[h, s, j] > 0.0)[0]:
                yield from walk(
                    h + 1,
                    states + [int(nxt)],
                    actions + [j],
                    p * pa * trans[h, s, j, nxt],
                )

    yield from walk(0, [game.s_init], [], 1.0)


def product_probs(policies):
    """``prod_j pi^j_h(a_j|s)`` as a ``(H, S, A)`` numpy array."""
    out = None
    for pol in policies:
        pol = np.asarray(pol)
        out = pol if out is None else out[..., :, None] * pol[..., None, :]
        out = out.reshape(out.shape[:2] + (-1,))
    return out


def expected_costs(game, policies):
    """``V^{i,pi}(s_1)`` for every player, from explicit enumeration."""
    costs = np.asarray(game.costs)
    total = np.zeros(game.n_players)
    for p, states, actions in enumerate_trajectories(game, policies):
        for h, (s, j) in enumerate(zip(states, actions)):
            total += p * costs[h, :, s, j]
    return total


def minimax_value(game, player=0):
    """Minimax value of a two-player constant-sum layered game at ``s_1``.

    Backward induction; at each ``(h, s)`` the one-shot matrix game
    ``M[a_1, a_2] = l^0_h(s, a) + sum_{s'} P_h(s'|s,a) V_{h+1}(s')`` is solved by
    the standard LP for the minimizing player. Independent of anything in
    ``algorithm.py``.
    """
    assert game.n_players == 2 and player == 0
    P = np.asarray(game.transitions)
    C = np.asarray(game.costs)
    n1, n2 = game.n_actions
    values = np.zeros(game.n_states)
    for h in range(game.horizon - 1, -1, -1):
        nxt = np.zeros(game.n_states)
        for s in range(game.n_states):
            payoff = (C[h, 0, s] + P[h, s] @ values).reshape(n1, n2)
            nxt[s] = _matrix_game_value(payoff)
        values = nxt
    return float(values[game.s_init])


def _matrix_game_value(payoff):
    """``min_x max_y x^T M y`` over the simplices, by linear programming."""
    n1, n2 = payoff.shape
    obj = np.zeros(n1 + 1)
    obj[-1] = 1.0
    a_ub = np.concatenate([payoff.T, -np.ones((n2, 1))], axis=1)
    a_eq = np.concatenate([np.ones((1, n1)), np.zeros((1, 1))], axis=1)
    res = linprog(
        obj,
        A_ub=a_ub,
        b_ub=np.zeros(n2),
        A_eq=a_eq,
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * n1 + [(None, None)],
    )
    if not res.success:
        raise RuntimeError(f"matrix game LP failed: {res.message}")
    return float(res.x[-1])


def joint_action_index(game, actions):
    """Flat C-order index of a per-player action tuple."""
    return int(np.ravel_multi_index(tuple(actions), game.n_actions))


def all_joint_actions(game):
    """Every joint action as a tuple, in flat-index order."""
    return list(itertools.product(*[range(n) for n in game.n_actions]))
