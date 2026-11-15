"""Tabular layered Markov games.

Notation follows Section 2.2 of arXiv:2609.00504. A game is
``(H, {S_h}, {A_i}, {P_h}, {l^i_h}, s_1)``: play runs for ``H`` layers, at layer
``h`` every player picks an action simultaneously, player ``i`` pays
``l^i_h(s, a)`` and the state moves by ``P_h(. | s, a)``.

States of different layers are stored in one index range of size ``n_states``.
A state that does not belong to layer ``h`` simply has zero occupancy there;
its rows in ``transitions`` and ``costs`` are filler and never read by anything
that starts from ``s_init``. Joint actions are flat indices into
``prod(n_actions)`` in C order, so joint index ``a`` unravels to
``(a_1, ..., a_m)``.

Costs live in ``[0, 1]`` as the paper assumes, which is what makes the
truncation error bound ``gamma^L / (1 - gamma)`` of Section 2.4 hold.
"""

from __future__ import annotations

import dataclasses
import itertools

import jax.numpy as jnp
import numpy as np

__all__ = [
    "MarkovGame",
    "matrix_game",
    "matching_pennies",
    "rock_paper_scissors",
    "battle_of_the_sexes_chain",
    "routing_game",
    "public_goods_game",
    "transition_trap_game",
    "zero_sum_chain",
    "StationaryGame",
    "random_stationary_zero_sum",
    "discounted_truncation",
    "benchmark_games",
]


@dataclasses.dataclass(frozen=True)
class MarkovGame:
    """A finite-horizon layered Markov game with tabular transitions and costs.

    Attributes:
        horizon: number of layers ``H``.
        n_actions: ``|A_i|`` for each player, length ``m``.
        transitions: ``P_h(s' | s, a)`` with shape ``(H, S, A, S)``.
        costs: ``l^i_h(s, a)`` with shape ``(H, m, S, A)``.
        s_init: the deterministic initial state ``s_1``.
    """

    horizon: int
    n_actions: tuple[int, ...]
    transitions: jnp.ndarray
    costs: jnp.ndarray
    s_init: int = 0

    @property
    def n_players(self) -> int:
        return len(self.n_actions)

    @property
    def n_states(self) -> int:
        return self.transitions.shape[1]

    @property
    def n_joint_actions(self) -> int:
        return int(np.prod(self.n_actions))

    def __post_init__(self):
        h, s, a, s2 = self.transitions.shape
        assert h == self.horizon and s == s2 and a == self.n_joint_actions
        assert self.costs.shape == (h, self.n_players, s, a)


def _build(horizon, n_actions, n_states, s_init, fill):
    """Assemble arrays from a python callable ``fill(h, s, joint_action)``.

    ``fill`` returns ``(next_state_probs, costs_per_player)`` or ``None`` for
    (layer, state) pairs that are not reachable, in which case the state absorbs
    into itself at zero cost.
    """
    m = len(n_actions)
    n_a = int(np.prod(n_actions))
    P = np.zeros((horizon, n_states, n_a, n_states))
    C = np.zeros((horizon, m, n_states, n_a))
    for h in range(horizon):
        for s in range(n_states):
            for j, act in enumerate(itertools.product(*[range(k) for k in n_actions])):
                out = fill(h, s, act)
                if out is None:
                    P[h, s, j, s] = 1.0
                    continue
                probs, cost = out
                for sp, p in probs.items():
                    P[h, s, j, sp] += p
                C[h, :, s, j] = cost
    assert np.allclose(P.sum(-1), 1.0)
    assert C.min() >= 0.0 and C.max() <= 1.0
    return MarkovGame(horizon, tuple(n_actions), jnp.asarray(P), jnp.asarray(C), s_init)


def matrix_game(cost_tensor) -> MarkovGame:
    """A one-layer Markov game, i.e. a normal-form game.

    ``cost_tensor`` has shape ``(m, |A_1|, ..., |A_m|)`` and holds each player's
    cost for every joint action.
    """
    cost_tensor = np.asarray(cost_tensor, dtype=float)
    m = cost_tensor.shape[0]
    n_actions = tuple(int(k) for k in cost_tensor.shape[1:])
    assert len(n_actions) == m
    n_a = int(np.prod(n_actions))
    P = np.ones((1, 1, n_a, 1))
    C = cost_tensor.reshape(m, n_a)[None, :, None, :]
    return MarkovGame(1, n_actions, jnp.asarray(P), jnp.asarray(C), 0)


def matching_pennies() -> MarkovGame:
    """Player 1 pays 1 when the pennies match. Unique equilibrium is uniform."""
    c1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    return matrix_game(np.stack([c1, 1.0 - c1]))


def rock_paper_scissors() -> MarkovGame:
    """Costs in {0, 0.5, 1}. Unique equilibrium is uniform."""
    win = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
    c1 = 0.5 * (1.0 - win)  # player 1 pays less when player 1 wins
    return matrix_game(np.stack([c1, 1.0 - c1]))


def battle_of_the_sexes_chain() -> MarkovGame:
    """Appendix F.2.1: two players, two actions, H = 2, states {0}, {1,2}, {3}."""
    first = {(0, 0): ({1: 0.9, 2: 0.1}, (0.05, 0.20)),
             (1, 1): ({1: 0.1, 2: 0.9}, (0.20, 0.05)),
             (0, 1): ({1: 0.5, 2: 0.5}, (0.55, 0.55)),
             (1, 0): ({1: 0.5, 2: 0.5}, (0.55, 0.55))}
    second = {1: {(0, 0): (0.15, 0.35), (1, 1): (0.35, 0.15),
                  (0, 1): (0.65, 0.65), (1, 0): (0.65, 0.65)},
              2: {(1, 1): (0.15, 0.35), (0, 0): (0.35, 0.15),
                  (0, 1): (0.65, 0.65), (1, 0): (0.65, 0.65)}}

    def fill(h, s, a):
        if h == 0 and s == 0:
            return first[a]
        if h == 1 and s in (1, 2):
            return {3: 1.0}, second[s][a]
        return None

    return _build(2, (2, 2), 4, 0, fill)


def routing_game() -> MarkovGame:
    """Appendix F.2.2: two players, two routes, H = 3, congestion when equal."""

    def fill(h, s, a):
        kappa = float(a[0] == a[1])
        n0 = (a[0] == 0) + (a[1] == 0)
        if h == 0 and s == 0:
            p1 = 0.75 if n0 <= 1 else 0.25
            cost = tuple(0.10 + 0.45 * kappa + 0.05 * a[i] for i in range(2))
            return {1: p1, 2: 1.0 - p1}, cost
        if h == 1 and s in (1, 2):
            if s == 1:
                p3 = 0.85 if kappa == 0 else 0.35
            else:
                p3 = 0.35 if kappa == 0 else 0.15
            base = 0.05 if s == 1 else 0.20
            cost = tuple(base + 0.50 * kappa + 0.05 * (a[i] == 0) for i in range(2))
            return {3: p3, 4: 1.0 - p3}, cost
        if h == 2 and s in (3, 4):
            pen = [0.10 if ((s == 3 and a[i] == 1) or (s == 4 and a[i] == 0)) else 0.0
                   for i in range(2)]
            cost = tuple(0.10 + 0.55 * kappa + pen[i] for i in range(2))
            return {5: 1.0}, cost
        return None

    return _build(3, (2, 2), 6, 0, fill)


def public_goods_game() -> MarkovGame:
    """Appendix F.2.3: three players, contribute or not, H = 2."""

    def fill(h, s, a):
        k = sum(a)
        if h == 0 and s == 0:
            p1 = 0.85 if k >= 2 else 0.20
            cost = tuple(0.08 + 0.22 * a[i] + 0.35 * max(0, 2 - k) / 2 for i in range(3))
            return {1: p1, 2: 1.0 - p1}, cost
        if h == 1 and s == 1:
            cost = tuple(0.10 + 0.18 * a[i] + 0.08 * (k < 1) for i in range(3))
            return {3: 1.0}, cost
        if h == 1 and s == 2:
            cost = tuple(0.55 + 0.15 * a[i] - 0.10 * min(k, 2) for i in range(3))
            return {3: 1.0}, cost
        return None

    return _build(2, (2, 2, 2), 4, 0, fill)


def transition_trap_game() -> MarkovGame:
    """Appendix F.2.4: action 0 is a cheap shortcut that raises the trap risk."""
    danger = {0: 0.10, 1: 0.50, 2: 0.80}
    trap = {1: {0: 0.05, 1: 0.20, 2: 0.50}, 2: {0: 0.10, 1: 0.50, 2: 0.80}}

    def fill(h, s, a):
        n0 = (a[0] == 0) + (a[1] == 0)
        if h == 0 and s == 0:
            p2 = danger[n0]
            cost = tuple(0.05 if a[i] == 0 else 0.13 for i in range(2))
            return {1: 1.0 - p2, 2: p2}, cost
        if h == 1 and s in (1, 2):
            p4 = trap[s][n0]
            extra = 0.02 if s == 2 else 0.0
            cost = tuple((0.06 if a[i] == 0 else 0.20) + extra for i in range(2))
            return {3: 1.0 - p4, 4: p4}, cost
        if h == 2 and s == 3:
            return {5: 1.0}, (0.20, 0.20)
        if h == 2 and s == 4:
            cost = tuple(0.80 if a[i] == 0 else 0.60 for i in range(2))
            return {5: 1.0}, cost
        return None

    return _build(3, (2, 2), 6, 0, fill)


def zero_sum_chain() -> MarkovGame:
    """A two-player constant-sum layered game, used as a minimax benchmark.

    Player 2's cost is ``1 - l^1``, so ``J^1 + J^2 = H`` and the game has a
    well defined minimax value that backward induction with a matrix-game LP at
    every state can compute independently of any learning dynamics.
    """
    first = {(0, 0): ({1: 0.8, 2: 0.2}, 0.30),
             (0, 1): ({1: 0.3, 2: 0.7}, 0.75),
             (1, 0): ({1: 0.6, 2: 0.4}, 0.80),
             (1, 1): ({1: 0.1, 2: 0.9}, 0.25)}
    second = {1: {(0, 0): 0.20, (0, 1): 0.70, (1, 0): 0.90, (1, 1): 0.35},
              2: {(0, 0): 0.85, (0, 1): 0.15, (1, 0): 0.40, (1, 1): 0.60}}

    def fill(h, s, a):
        if h == 0 and s == 0:
            probs, c = first[a]
            return probs, (c, 1.0 - c)
        if h == 1 and s in (1, 2):
            c = second[s][a]
            return {3: 1.0}, (c, 1.0 - c)
        return None

    return _build(2, (2, 2), 4, 0, fill)


@dataclasses.dataclass(frozen=True)
class StationaryGame:
    """An infinite-horizon discounted game with a stationary kernel and costs."""

    n_actions: tuple[int, ...]
    transitions: jnp.ndarray  # (S, A, S)
    costs: jnp.ndarray  # (m, S, A)
    discount: float
    s_init: int = 0


def random_stationary_zero_sum(key, n_states=3, n_actions=(2, 2), discount=0.8):
    """A random two-player constant-sum stationary discounted game."""
    import jax

    k1, k2 = jax.random.split(key)
    n_a = int(np.prod(n_actions))
    logits = jax.random.normal(k1, (n_states, n_a, n_states))
    P = jax.nn.softmax(logits, axis=-1)
    c1 = jax.random.uniform(k2, (n_states, n_a))
    C = jnp.stack([c1, 1.0 - c1])
    return StationaryGame(tuple(n_actions), P, C, discount, 0)


def discounted_truncation(game: StationaryGame, horizon: int) -> MarkovGame:
    """The ``L``-step discounted truncation ``G^[L]_{gamma,Hbar}`` of Section 2.4.

    One-step costs become ``c^i_h(s, a) = gamma^{h-1} l^i(s, a)`` for
    ``h in [L]``; the kernel, the action sets and the initial state are
    inherited. Running the finite-horizon algorithm on this game and extending
    the output by any continuation costs at most ``gamma^L / (1 - gamma)``
    extra in the untruncated objective (Theorem 3).
    """
    P = jnp.broadcast_to(game.transitions[None], (horizon,) + game.transitions.shape)
    disc = game.discount ** jnp.arange(horizon)
    C = disc[:, None, None, None] * game.costs[None]
    return MarkovGame(horizon, game.n_actions, P, C, game.s_init)


def benchmark_games() -> dict[str, MarkovGame]:
    """The four finite-horizon games of Section 6.1."""
    return {
        "battle_of_the_sexes_chain": battle_of_the_sexes_chain(),
        "routing": routing_game(),
        "public_goods": public_goods_game(),
        "transition_trap": transition_trap_game(),
    }
