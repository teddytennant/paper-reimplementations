"""Policy evaluation, induced MDPs and best responses."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from layered_omd import games, reference, values


def _random_policies(game, seed):
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.dirichlet(np.ones(n), size=(game.horizon, game.n_states)))
        for n in game.n_actions
    )


ALL_GAMES = list(games.benchmark_games().items()) + [
    ("zero_sum_chain", games.zero_sum_chain()),
    ("matching_pennies", games.matching_pennies()),
]


@pytest.mark.parametrize("name,game", ALL_GAMES)
def test_backward_induction_matches_trajectory_enumeration(name, game):
    """V^{i,pi}(s_1) from the scan equals the sum over every trajectory."""
    for seed in range(3):
        policies = _random_policies(game, seed)
        exact = reference.expected_costs(game, policies)
        ours = np.array([float(values.value(game, policies, i)) for i in range(game.n_players)])
        assert np.max(np.abs(ours - exact)) < 1e-12


@pytest.mark.parametrize("name,game", ALL_GAMES)
def test_value_is_the_policy_average_of_q(name, game):
    """V^{i,pi}_h(s) = sum_a pi^i_h(a|s) Q^{i,pi}_h(s, a), by definition."""
    policies = _random_policies(game, 7)
    for i in range(game.n_players):
        q, v = values.player_values(game, policies, i)
        recon = jnp.einsum("hsa,hsa->hs", policies[i], q)
        assert np.max(np.abs(np.asarray(recon - v[:-1]))) < 1e-13
        assert np.max(np.abs(np.asarray(v[-1]))) == 0.0


def test_discounted_truncation_matches_the_exact_linear_system():
    """(I - gamma P_pi)^{-1} r_pi against L layers of c_h = gamma^{h-1} l."""
    key = jax.random.key(0)
    stationary = games.random_stationary_zero_sum(key, n_states=4, discount=0.85)
    rng = np.random.default_rng(11)
    policies_1step = tuple(
        jnp.asarray(rng.dirichlet(np.ones(n), size=(1, 4))) for n in stationary.n_actions
    )
    joint = reference.product_probs(policies_1step)[0]  # (S, A)
    trans = np.asarray(stationary.transitions)
    costs = np.asarray(stationary.costs)
    gamma = stationary.discount

    for player in range(2):
        p_pi = np.einsum("sa,sax->sx", joint, trans)
        r_pi = np.einsum("sa,sa->s", joint, costs[player])
        exact = np.linalg.solve(np.eye(4) - gamma * p_pi, r_pi)[stationary.s_init]

        previous = -np.inf
        for horizon in (1, 5, 20, 80, 300):
            game = games.discounted_truncation(stationary, horizon)
            stacked = tuple(jnp.tile(p, (horizon, 1, 1)) for p in policies_1step)
            got = float(values.value(game, stacked, player))
            tail = gamma ** horizon / (1.0 - gamma)
            # costs are nonnegative, so truncation can only lose value, by at
            # most the discounted tail (Section 2.4)
            assert -1e-12 <= exact - got <= tail + 1e-12
            assert got > previous
            previous = got
        assert abs(exact - got) < 1e-12


def test_best_response_beats_every_deterministic_markov_policy():
    """Brute force over deterministic policies in the induced MDP."""
    game = games.battle_of_the_sexes_chain()
    policies = _random_policies(game, 3)
    for player in range(2):
        br_value, br_policy = values.best_response(game, policies, player)
        n = game.n_actions[player]
        cells = [(h, s) for h in range(game.horizon) for s in range(game.n_states)]
        tables = []
        for choice in itertools.product(range(n), repeat=len(cells)):
            table = np.zeros((game.horizon, game.n_states, n))
            for (h, s), a in zip(cells, choice):
                table[h, s, a] = 1.0
            tables.append(table)
        stacked = jnp.asarray(np.stack(tables))

        def one(table, player=player):
            deviated = list(policies)
            deviated[player] = table
            return values.value(game, tuple(deviated), player)

        best = float(np.min(np.asarray(jax.vmap(one)(stacked))))
        assert abs(float(br_value) - best) < 1e-12
        # the greedy table returned is one of the minimizers
        deviated = list(policies)
        deviated[player] = br_policy
        assert abs(float(values.value(game, tuple(deviated), player)) - best) < 1e-12


@pytest.mark.parametrize("name,game", ALL_GAMES)
def test_best_response_is_never_worse_than_playing_along(name, game):
    for seed in range(3):
        policies = _random_policies(game, 100 + seed)
        for player in range(game.n_players):
            br = float(values.best_response(game, policies, player)[0])
            assert br <= float(values.value(game, policies, player)) + 1e-13


def test_induced_mdp_reproduces_the_game_costs_and_kernel():
    """l^i_t and P^i_t are the a_{-i} averages of the game's own tables."""
    game = games.public_goods_game()
    policies = _random_policies(game, 5)
    ell = np.asarray(values.induced_costs(game, policies, 1))
    trans = np.asarray(values.induced_transitions(game, policies, 1))
    costs = np.asarray(game.costs)
    kernel = np.asarray(game.transitions)
    pol = [np.asarray(p) for p in policies]
    for h in range(game.horizon):
        for s in range(game.n_states):
            for own in range(game.n_actions[1]):
                acc_c, acc_p = 0.0, np.zeros(game.n_states)
                for a0, a2 in itertools.product(range(2), repeat=2):
                    weight = pol[0][h, s, a0] * pol[2][h, s, a2]
                    j = reference.joint_action_index(game, (a0, own, a2))
                    acc_c += weight * costs[h, 1, s, j]
                    acc_p += weight * kernel[h, s, j]
                assert abs(acc_c - ell[h, s, own]) < 1e-13
                assert np.max(np.abs(acc_p - trans[h, s, own])) < 1e-13
