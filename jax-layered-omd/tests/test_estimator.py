"""The blocked trajectory estimator of Section 3.2."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from layered_omd import algorithm, games, reference, values


def _floored_policies(game, seed, zeta=0.2):
    """A profile inside Delta^zeta, so every (s, a_i) really is visited."""
    rng = np.random.default_rng(seed)
    out = []
    for n in game.n_actions:
        raw = rng.dirichlet(np.ones(n), size=(game.horizon, game.n_states))
        out.append(jnp.asarray((1.0 - zeta) * raw + zeta / n))
    return tuple(out)


@pytest.mark.parametrize(
    "name,game",
    [
        ("battle_of_the_sexes_chain", games.battle_of_the_sexes_chain()),
        ("transition_trap", games.transition_trap_game()),
        ("public_goods", games.public_goods_game()),
    ],
)
def test_tail_cost_is_conditionally_unbiased_for_q(name, game):
    """E[Y^i_h 1{s_h = s, a^i_h = a_i}] = Pr(s_h = s, a^i_h = a_i) Q^{i,pi}_h(s, a_i).

    Checked by enumerating every trajectory, so this is an exact identity to
    machine precision, not a sampling statement. It is what makes the block
    estimate ``Qhat`` unbiased for ``Q`` conditional on the pair being visited;
    the unconditional estimator is low by exactly ``Q Pr(N = 0)`` because it
    returns 0 when the pair is never seen.
    """
    policies = _floored_policies(game, 0)
    costs = np.asarray(game.costs)
    joint_tuples = reference.all_joint_actions(game)

    numer = np.zeros((game.n_players, game.horizon, game.n_states, max(game.n_actions)))
    denom = np.zeros_like(numer)
    for prob, states, actions in reference.enumerate_trajectories(game, policies):
        tail = np.zeros((game.n_players, game.horizon + 1))
        for h in reversed(range(game.horizon)):
            tail[:, h] = tail[:, h + 1] + costs[h, :, states[h], actions[h]]
        for h in range(game.horizon):
            own = joint_tuples[actions[h]]
            for i in range(game.n_players):
                numer[i, h, states[h], own[i]] += prob * tail[i, h]
                denom[i, h, states[h], own[i]] += prob

    for i in range(game.n_players):
        q_exact = np.asarray(values.player_values(game, policies, i)[0])
        seen = denom[i, :, :, : game.n_actions[i]] > 1e-14
        implied = numer[i, :, :, : game.n_actions[i]][seen] / denom[
            i, :, :, : game.n_actions[i]
        ][seen]
        assert seen.sum() > 0
        assert np.max(np.abs(implied - q_exact[seen])) < 1e-12


def test_estimate_q_computes_the_written_formula():
    """Recompute Qhat from the raw rollout in numpy and compare."""
    game = games.transition_trap_game()
    policies = _floored_policies(game, 1)
    rollout = algorithm.sample_block(game, policies, 64, jax.random.key(3))
    states, actions, tail = (np.asarray(x) for x in rollout)
    ours = [np.asarray(q) for q in algorithm.estimate_q(game, rollout)]
    for i in range(game.n_players):
        expected = np.zeros((game.horizon, game.n_states, game.n_actions[i]))
        for h in range(game.horizon):
            for s in range(game.n_states):
                for a in range(game.n_actions[i]):
                    hit = (states[h] == s) & (actions[i, h] == a)
                    if hit.sum() > 0:
                        expected[h, s, a] = tail[i, h][hit].mean()
        assert np.max(np.abs(ours[i] - expected)) < 1e-12


def test_block_estimate_converges_to_the_exact_q():
    """Monte Carlo: the block estimate matches Q^{i,pi} at the rate 1/sqrt(B).

    The tolerance is a measured multiple of the standard error rather than a
    round number: with ``B`` episodes a pair reached with probability ``p`` is
    seen about ``pB`` times and the tail cost is bounded by ``H``, so the error
    of a well visited pair is about ``H / sqrt(pB)``.
    """
    game = games.transition_trap_game()
    policies = _floored_policies(game, 2, zeta=0.4)
    exact = [np.asarray(values.player_values(game, policies, i)[0]) for i in range(2)]

    errors = {}
    for block in (500, 8000):
        per_seed = []
        for seed in range(5):
            rollout = algorithm.sample_block(game, policies, block, jax.random.key(seed))
            states, actions, _ = (np.asarray(x) for x in rollout)
            est = [np.asarray(q) for q in algorithm.estimate_q(game, rollout)]
            worst = 0.0
            for i in range(2):
                counts = np.zeros_like(exact[i])
                for h in range(game.horizon):
                    for s in range(game.n_states):
                        for a in range(game.n_actions[i]):
                            counts[h, s, a] = ((states[h] == s) & (actions[i, h] == a)).sum()
                mask = counts >= 0.02 * block
                worst = max(worst, float(np.max(np.abs(est[i] - exact[i])[mask])))
            per_seed.append(worst)
        errors[block] = float(np.median(per_seed))

    # H = 3 and the mask keeps pairs with at least 0.02 B visits, so
    # H / sqrt(0.02 B) is the scale to beat.
    for block, err in errors.items():
        assert err < 4.0 * game.horizon / np.sqrt(0.02 * block)
    assert errors[8000] < 0.5 * errors[500]


def test_sampler_visit_frequencies_match_the_model():
    """State-layer occupancies from the sampler match the exact occupancies."""
    game = games.routing_game()
    policies = _floored_policies(game, 4, zeta=0.3)
    joint = reference.product_probs(policies)
    trans = np.asarray(game.transitions)
    occupancy = np.zeros((game.horizon, game.n_states))
    dist = np.zeros(game.n_states)
    dist[game.s_init] = 1.0
    for h in range(game.horizon):
        occupancy[h] = dist
        dist = np.einsum("s,sa,sax->x", dist, joint[h], trans[h])

    states, _, _ = algorithm.sample_block(game, policies, 40000, jax.random.key(9))
    states = np.asarray(states)
    empirical = np.stack(
        [np.bincount(states[h], minlength=game.n_states) / states.shape[1]
         for h in range(game.horizon)]
    )
    # 4 standard errors at n = 40000 is about 0.008
    assert np.max(np.abs(empirical - occupancy)) < 0.01
