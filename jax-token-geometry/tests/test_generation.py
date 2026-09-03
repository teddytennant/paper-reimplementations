"""Section 4.1 and Theorem 4: what accurate prediction buys a generator.

Two things are separated here. The sharp statement, that the off-support mass a
generator places under the pre-training context distribution is at most
epsilon_p, is checked with the constant equal to one. Theorem 4 as stated
measures the same quantity under the generator's own context distribution and
pays the density ratio C_{Q_q} of Assumption 3.6, which is enormous in this
simulation: KL(p; q) does not penalize mass that q puts where p is zero, so the
fitted generator reaches contexts that the pre-training distribution reaches
only very rarely. The inequality still holds, and it is checked, but the
informative version is the one under the pre-training measure.
"""

import numpy as np
import pytest

from tokengeom import downstream as D
from tokengeom import geometry as G

REWARD_SCALE = 1.0
REWARD_ERROR = 0.05


def support_mass(q, mask, weights):
    return float(np.asarray(weights) @ (np.asarray(q) * np.asarray(mask)).sum(axis=1))


def test_tilted_generator_solves_the_regularized_problem(sim, token_enum):
    """q propto pi exp(rhat / lambda) maximizes reward minus lambda KL(q; pi).

    The closed form of Section 4.1, checked against random feasible directions
    on a concave objective.
    """
    rng = np.random.default_rng(0)
    mask = D.plausible_mask(token_enum, sim)
    reward = D.make_reward(mask, seed=1, scale=REWARD_SCALE)
    reference = rng.random(reward.shape) + 0.05
    reference /= reference.sum(axis=1, keepdims=True)
    lam = 0.3
    optimal = D.tilted_generator(reference, reward, lam)
    rows = rng.choice(len(reward), size=40, replace=False)

    def objective(q):
        kl = np.sum(q * (np.log(q) - np.log(reference[rows])), axis=1)
        return (q * reward[rows]).sum(axis=1) - lam * kl

    best = objective(optimal[rows])
    for _ in range(20):
        step = rng.standard_normal(optimal[rows].shape) * 0.02
        candidate = np.maximum(optimal[rows] + step - step.mean(axis=1, keepdims=True), 1e-12)
        candidate /= candidate.sum(axis=1, keepdims=True)
        assert np.all(objective(candidate) <= best + 1e-12)


def test_off_support_mass_is_bounded_by_the_prediction_error(encoder, sim, token_enum):
    """1 - rho_p <= epsilon_p under the pre-training context distribution.

    This is Theorem 4's first claim with C_{Q} = 1, which is what the change of
    measure in the theorem reduces to when the generator is scored on the
    distribution it was fitted on. It follows from KL(p; q) >= -log q(S).
    """
    _, _, pair = encoder
    q = np.asarray(np.exp(pair.log_predictions))
    mask = D.plausible_mask(token_enum, sim)
    on_support = support_mass(q, mask, token_enum.context_prob)
    assert 1.0 - on_support <= pair.epsilon_p
    assert on_support > 0.75


def test_plausibility_improves_along_the_training_path(fit_path, sim, semantic_enum):
    """Figure 4 top left: less prediction error, more mass on S_j(X)."""
    mask = D.plausible_mask(semantic_enum, sim)
    errors, masses = [], []
    for pair in fit_path:
        q = np.asarray(np.exp(pair.log_predictions))
        errors.append(pair.epsilon_p)
        masses.append(support_mass(q, mask, semantic_enum.context_prob))
        assert 1.0 - masses[-1] <= pair.epsilon_p + 1e-12
    assert G.spearman(errors, masses) < -0.9
    assert masses[int(np.argmin(errors))] > 0.999


def test_theorem4_plausibility_bounds(encoder, sim, token_enum):
    """Both plausibility bounds of Theorem 4, with C_{Q_q} measured exactly.

    The density ratio is computed from the stationary context distribution of
    the chain each generator drives, so no constant is assumed.
    """
    _, _, pair = encoder
    d = sim.config.vocab_size
    mask = D.plausible_mask(token_enum, sim)
    base = np.asarray(token_enum.context_prob)

    q_pre = np.asarray(np.exp(pair.log_predictions))
    ctx_pre = D.generator_context_distribution(q_pre, d)
    coverage_pre = D.density_ratio(ctx_pre, base)
    assert np.isfinite(coverage_pre)
    rho_pre = D.plausibility(q_pre, mask, ctx_pre)
    assert rho_pre >= D.theorem4_plausibility_pre(coverage_pre, pair.epsilon_p)

    lam = 0.2
    reward = D.make_reward(mask, seed=2, scale=REWARD_SCALE)
    reward_hat = D.estimate_reward(reward, REWARD_ERROR, seed=3)
    q_post = D.tilted_generator(q_pre, reward_hat, lam)
    ctx_post = D.generator_context_distribution(q_post, d)
    coverage_post = D.density_ratio(ctx_post, base)
    rho_post = D.plausibility(q_post, mask, ctx_post)
    assert rho_post >= D.theorem4_plausibility_post(
        coverage_post, pair.epsilon_p, REWARD_ERROR, lam
    )
    # post-training inherits the plausibility of its reference
    assert rho_post > 0.9


def test_theorem4_regret_bounds(encoder, sim, token_enum):
    """The regret bounds of Theorem 4 for both reference distributions."""
    _, _, pair = encoder
    d = sim.config.vocab_size
    k = sim.sparsity_k
    mask = D.plausible_mask(token_enum, sim)
    base = np.asarray(token_enum.context_prob)
    reward = D.make_reward(mask, seed=4, scale=REWARD_SCALE)
    reward_hat = D.estimate_reward(reward, REWARD_ERROR, seed=5)
    q_pre = np.asarray(np.exp(pair.log_predictions))

    for lam in (0.1, 0.2, 0.5):
        q_post = D.tilted_generator(q_pre, reward_hat, lam)
        ctx_post = D.generator_context_distribution(q_post, d)
        coverage = D.density_ratio(ctx_post, base)
        regret = D.reward_regret(q_post, reward, ctx_post)
        assert regret <= D.theorem4_regret_post(
            REWARD_SCALE, REWARD_ERROR, lam, k, coverage, pair.epsilon_p
        ) + 1e-12

        uniform = np.full_like(q_pre, 1.0 / d)
        q_uniform = D.tilted_generator(uniform, reward_hat, lam)
        ctx_uniform = D.generator_context_distribution(q_uniform, d)
        assert D.reward_regret(q_uniform, reward, ctx_uniform) <= (
            D.theorem4_regret_uniform(REWARD_SCALE, REWARD_ERROR, lam, d) + 1e-12
        )


def test_pretrained_reference_shrinks_the_choice_set(encoder, sim, token_enum):
    """Figure 4 bottom: the pre-trained reference against a uniform one.

    The two upper bounds of Theorem 4 differ by lambda log k against
    lambda log d, so the advantage should grow with the temperature. It does,
    and the pre-trained reference wins at the two larger temperatures. Section
    4.1 is explicit that the comparison is a worst-case one and need not hold
    for every instance, which is what the smallest temperature shows here: at
    lambda = 0.1 the reward tilt dominates and the reference barely matters.
    """
    _, _, pair = encoder
    d = sim.config.vocab_size
    mask = D.plausible_mask(token_enum, sim)
    reward = D.make_reward(mask, seed=6, scale=REWARD_SCALE)
    reward_hat = D.estimate_reward(reward, REWARD_ERROR, seed=7)
    q_pre = np.asarray(np.exp(pair.log_predictions))
    uniform = np.full_like(q_pre, 1.0 / d)

    gaps = []
    for lam in (0.1, 0.2, 0.5):
        q_post = D.tilted_generator(q_pre, reward_hat, lam)
        q_uniform = D.tilted_generator(uniform, reward_hat, lam)
        regret_post = D.reward_regret(
            q_post, reward, D.generator_context_distribution(q_post, d)
        )
        regret_uniform = D.reward_regret(
            q_uniform, reward, D.generator_context_distribution(q_uniform, d)
        )
        gaps.append(regret_uniform - regret_post)
    assert np.all(np.diff(gaps) > 0)
    assert gaps[-1] > 0.0
    assert gaps[-2] > 0.0


def test_reward_construction_satisfies_assumption3(sim, semantic_enum):
    """Assumption 3.1 to 3.3: the reward favors the plausible set, is bounded,
    and the estimate is uniformly accurate."""
    mask = D.plausible_mask(semantic_enum, sim)
    reward = D.make_reward(mask, seed=8, scale=REWARD_SCALE)
    assert reward.min() >= 0.0
    assert reward.max() <= REWARD_SCALE
    inside = np.where(mask, reward, np.inf).min(axis=1)
    outside = np.where(mask, -np.inf, reward).max(axis=1)
    assert np.all(inside - outside >= 0.0)
    assert np.all(np.where(mask, reward, 0.0).max(axis=1) > outside)
    hat = D.estimate_reward(reward, REWARD_ERROR, seed=9)
    assert np.abs(hat - reward).max() == pytest.approx(REWARD_ERROR)
