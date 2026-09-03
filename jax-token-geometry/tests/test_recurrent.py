"""Section 3: self-consistency, contraction, and recurrent-depth refinement.

Theorem 2 says that if the self-consistency operator is a kappa-contraction and
the block approximates it within delta, then

    D(Psi^L, Theta) <= kappa^L D(Psi^0, Theta) + (1 - kappa^L)/(1 - kappa) delta.

That inequality is checked on the finite operator built in tokengeom.recurrent,
where kappa and delta are both known by construction and verified empirically.
The trained encoder is checked against the empirical claim of Figure 3 instead,
because a residual transformer block is not a contraction in this metric.
"""

import numpy as np
import pytest

from tokengeom import geometry as G
from tokengeom import model as M
from tokengeom import recurrent as R


def test_context_conditionals_are_averaging_operators(sim):
    """P(x_{j-4+i} = . | x_j = v) is stochastic and preserves p(v).

    Preserving the marginal is what makes it nonexpansive in the metric
    D(Psi, Phi) = {E_{X,j} ||Psi_j - Phi_j||^2}^{1/2}, and so what turns the
    Lipschitz condition of Section 3.2 into a contraction.
    """
    gathers = np.asarray(R.context_conditionals(sim))
    assert np.allclose(gathers.sum(axis=2), 1.0, atol=1e-12)
    assert gathers.min() >= 0.0
    for block in gathers:
        assert np.abs(sim.token_marginal @ block - sim.token_marginal).max() < 1e-11


def test_averaging_operators_are_nonexpansive(sim):
    rng = np.random.default_rng(0)
    gathers = R.context_conditionals(sim)
    weight = sim.token_marginal
    for _ in range(20):
        psi = rng.standard_normal((sim.config.vocab_size, 5))
        base = R.map_distance(psi, np.zeros_like(psi), weight)
        for block in np.asarray(gathers):
            moved = block @ psi
            assert R.map_distance(moved, np.zeros_like(moved), weight) <= base + 1e-12


def test_certified_contraction_factor_bounds_the_empirical_one(sim):
    """kappa_Theta = sqrt(L_g) really is an upper bound on the operator's
    Lipschitz constant in D."""
    gathers = R.context_conditionals(sim)
    weight = sim.token_marginal
    for kappa in (0.2, 0.5, 0.8, 0.95):
        block = R.make_block(seed=1, width=6, kappa=kappa)
        assert block.kappa == pytest.approx(kappa)
        empirical = R.empirical_lipschitz(block, gathers, weight, seed=2, n_trials=40)
        assert empirical <= kappa + 1e-12


def test_fixed_point_is_self_consistent(sim):
    """Theta = T_{g_Theta}(Theta), the self-consistency principle of Section 3.1."""
    gathers = R.context_conditionals(sim)
    weight = sim.token_marginal
    rng = np.random.default_rng(3)
    for kappa in (0.3, 0.7, 0.9):
        block = R.make_block(seed=4, width=6, kappa=kappa)
        start = rng.standard_normal((sim.config.vocab_size, 6))
        theta = R.fixed_point(block, gathers, start, weight)
        assert R.map_distance(R.apply_operator(block, gathers, theta), theta, weight) < 1e-13
        # reached from a different start, so the fixed point is unique
        other = R.fixed_point(block, gathers, 5.0 * rng.standard_normal(theta.shape), weight)
        assert R.map_distance(theta, other, weight) < 1e-11


def test_theorem2_exact_operator(sim):
    """delta = 0: repeated application converges at rate kappa^L.

    Randomized over the contraction factor, the width, the seed and the depth.
    """
    gathers = R.context_conditionals(sim)
    weight = sim.token_marginal
    rng = np.random.default_rng(5)
    for trial in range(12):
        kappa = float(rng.uniform(0.2, 0.95))
        width = int(rng.integers(3, 9))
        block = R.make_block(seed=trial, width=width, kappa=kappa)
        start = rng.standard_normal((sim.config.vocab_size, width))
        theta = R.fixed_point(block, gathers, start, weight)
        other = rng.standard_normal((sim.config.vocab_size, width))
        d0 = R.map_distance(other, theta, weight)
        iterates = R.iterate_operator(block, gathers, other, 10)
        for depth in range(1, 11):
            got = R.map_distance(iterates[depth], theta, weight)
            assert got <= R.theorem2_bound(kappa, 0.0, depth, d0) + 1e-12


def test_theorem2_with_block_approximation_error(sim):
    """The full bound, with a block that misses g_Theta by exactly delta.

    The perturbation is a fixed field whose D-norm is delta, so
    sup_Psi D(T_{g_Theta}(Psi), T_g(Psi)) = delta and the second term of
    Theorem 2 is known, not estimated.
    """
    gathers = R.context_conditionals(sim)
    weight = sim.token_marginal
    d = sim.config.vocab_size
    rng = np.random.default_rng(6)
    saw_informative = False
    for trial in range(12):
        kappa = float(rng.uniform(0.2, 0.9))
        delta = float(rng.uniform(0.01, 0.3))
        width = int(rng.integers(3, 9))
        exact = R.make_block(seed=trial, width=width, kappa=kappa)
        approx = R.make_block(seed=trial, width=width, kappa=kappa, delta=delta,
                              n_tokens=d, token_weight=weight)
        start = rng.standard_normal((d, width))
        theta = R.fixed_point(exact, gathers, start, weight)

        probe = rng.standard_normal((d, width))
        assert R.map_distance(
            R.apply_operator(exact, gathers, probe),
            R.apply_operator(approx, gathers, probe),
            weight,
        ) == pytest.approx(delta, rel=1e-10)

        other = rng.standard_normal((d, width))
        d0 = R.map_distance(other, theta, weight)
        iterates = R.iterate_operator(approx, gathers, other, 12)
        for depth in range(1, 13):
            bound = R.theorem2_bound(kappa, delta, depth, d0)
            assert R.map_distance(iterates[depth], theta, weight) <= bound + 1e-12
            if bound < d0:
                saw_informative = True
        # the approximation floor of Theorem 2
        assert R.map_distance(iterates[-1], theta, weight) <= delta / (1 - kappa) + 1e-12
    assert saw_informative


def test_recurrent_depth_bound_decreases_toward_the_floor(sim):
    """D_0 > D_infinity is the regime where more depth helps, Section 3.2."""
    kappa, delta, d0 = 0.7, 0.05, 3.0
    bounds = [R.theorem2_bound(kappa, delta, depth, d0) for depth in range(1, 61)]
    assert np.all(np.diff(bounds) < 0)
    assert bounds[-1] == pytest.approx(delta / (1 - kappa), rel=1e-6)


def test_ill_conditioned_coordinates_break_the_contraction(sim):
    """Section 3.1: g_Atilde(y) = A g_Theta(A^{-1} y) inflates the Lipschitz
    constant by up to ||A|| ||A^{-1}||, while the prediction loss cannot see it.

    So among predictively equivalent representations, the recurrent construction
    prefers the well-conditioned ones.
    """
    gathers = R.context_conditionals(sim)
    weight = sim.token_marginal
    width = 6
    kappa = 0.6
    block = R.make_block(seed=7, width=width, kappa=kappa)
    rng = np.random.default_rng(8)

    def conjugated_lipschitz(matrix):
        inverse = np.linalg.inv(matrix)
        worst = 0.0
        for _ in range(40):
            psi = rng.standard_normal((sim.config.vocab_size, width))
            phi = rng.standard_normal((sim.config.vocab_size, width))
            out_a = R.apply_operator(block, gathers, psi @ inverse.T) @ matrix.T
            out_b = R.apply_operator(block, gathers, phi @ inverse.T) @ matrix.T
            worst = max(
                worst,
                R.map_distance(out_a, out_b, weight) / R.map_distance(psi, phi, weight),
            )
        return worst

    factors, conditions = [], []
    for stretch in (1.0, 4.0, 20.0, 100.0):
        matrix = np.diag([stretch] + [1.0] * (width - 1))
        conditions.append(np.linalg.cond(matrix))
        factors.append(conjugated_lipschitz(matrix))
        assert factors[-1] <= conditions[-1] * kappa + 1e-9
    assert np.all(np.diff(factors) > 0)
    assert factors[0] < 1.0 < factors[-1]


# -- the trained encoder ------------------------------------------------------


def test_recurrent_depth_shares_its_parameters(sim):
    """A block applied L times has the parameter count of one block, Section 3.2."""
    from tests.conftest import encoder_config

    counts = {
        depth: M.parameter_count(M.init_params(encoder_config(sim, depth=depth), 0))
        for depth in (1, 2, 4, 8)
    }
    assert len(set(counts.values())) == 1
    deeper = M.parameter_count(
        M.init_params(encoder_config(sim, depth=1, n_sub_layers=2), 0)
    )
    assert deeper > counts[1]


def test_initial_representation_ignores_the_context_at_the_query_slot(sim, token_enum):
    """Psi^0_j(X) does not depend on X, which is why Figure 3 omits l = 0."""
    from tests.conftest import encoder_config

    config = encoder_config(sim)
    params = M.init_params(config, 0)
    psi0 = M.initial_representation(params, token_enum.contexts[:64], config)
    query = np.asarray(psi0[:, config.context_length])
    assert np.abs(query - query[0]).max() < 1e-15


def test_repeated_block_refines_toward_the_decoder_optimal_representation(
    encoder, ideal, token_enum
):
    """D_Z(l) falls geometrically along the shared block, Figure 3 right.

    The paper reports a fall by a factor of 3.9 over l = 1, ..., 8, close to a
    constant ratio of 0.82 per layer.
    """
    params, config, _ = encoder
    layers = R.encoder_layer_representations(params, token_enum, config)
    distances = R.normalized_distance_to_ideal(layers, ideal, token_enum.context_prob)
    assert len(distances) == config.depth + 1
    tail = distances[1:]
    assert np.all(np.diff(tail) < 0)
    assert tail[0] / tail[-1] > 2.0
    ratio = (tail[-1] / tail[0]) ** (1.0 / (len(tail) - 1))
    assert 0.6 < ratio < 0.95


def test_ideal_representation_is_the_constrained_minimizer(encoder, ideal):
    """Z_j(X) = argmin_{||Z|| <= 4} L(p_j(X); p_hat_Ehat(Z)).

    Checked against the norm constraint and against random feasible directions,
    on a convex objective.
    """
    import jax
    import jax.numpy as jnp

    _, _, pair = encoder
    norms = np.linalg.norm(np.asarray(ideal), axis=1)
    assert norms.max() <= M.REPRESENTATION_NORM + 1e-9

    def objective(z):
        logq = jax.nn.log_softmax(z @ pair.embed.T + pair.bias, axis=-1)
        return float(-jnp.sum(pair.enum.context_prob[:, None] * pair.enum.conditional * logq))

    base = objective(ideal)
    rng = np.random.default_rng(9)
    for _ in range(10):
        direction = rng.standard_normal(np.shape(ideal))
        candidate = np.asarray(ideal) + 0.05 * direction
        scale = np.minimum(
            1.0, M.REPRESENTATION_NORM / np.linalg.norm(candidate, axis=1, keepdims=True)
        )
        assert objective(jnp.asarray(candidate * scale)) >= base - 1e-9


def test_encoder_reaches_a_useful_prediction_error(encoder):
    """The paper's transformer fits land at epsilon_p between 0.074 and 0.144."""
    _, _, pair = encoder
    assert pair.epsilon_p < 0.45
    assert np.linalg.norm(np.asarray(pair.theta), axis=1).max() == pytest.approx(
        M.REPRESENTATION_NORM, rel=1e-10
    )
