"""PSO mechanics: Eqs. 4 and 5, Eqs. 2 and 3, and convergence."""

import numpy as np

from mfspnet.encoding import SearchSpace, decode
from mfspnet.pso import (
    COGNITIVE,
    INERTIA,
    SOCIAL,
    PSOConfig,
    initial_state,
    minimize,
    position_update,
    step,
    update_bests,
    velocity_update,
)

SPACE = SearchSpace()


def test_table_2_pso_constants():
    assert (INERTIA, COGNITIVE, SOCIAL) == (0.83, 1.17, 1.69)
    cfg = PSOConfig()
    assert (cfg.inertia, cfg.cognitive, cfg.social) == (0.83, 1.17, 1.69)


def test_velocity_update_is_equation_4_term_by_term():
    cfg = PSOConfig()
    v = np.array([[1.0, -2.0]])
    x = np.array([[10.0, 20.0]])
    p = np.array([[12.0, 18.0]])
    g = np.array([15.0, 25.0])
    r1 = np.array([[0.3, 0.7]])
    r2 = np.array([[0.6, 0.2]])
    expected = (
        cfg.inertia * v + cfg.cognitive * r1 * (p - x) + cfg.social * r2 * (g[None] - x)
    )
    assert np.allclose(velocity_update(v, x, p, g, r1, r2, cfg), expected)


def test_position_update_is_equation_5():
    cfg = PSOConfig(clamp_position=False)
    x = np.array([[10.0, 20.0]])
    v = np.array([[1.5, -3.0]])
    new_x, new_v = position_update(x, v, SPACE, cfg)
    assert np.allclose(new_x, np.array([[11.5, 17.0]]))
    assert np.allclose(new_v, v)


def test_zero_inertia_and_zero_social_moves_the_particle_toward_its_own_best():
    """Eq. 4 with w = c2 = 0 reduces to a step of size c1*rho1 toward pbest."""
    cfg = PSOConfig(inertia=0.0, social=0.0, cognitive=1.0)
    rng = np.random.default_rng(0)
    x = np.array([[12.0, 30.0, 20.0]])
    p = np.array([[20.0, 14.0, 20.0]])
    space = SearchSpace(l_max=3)
    state = initial_state(x, np.array([1.0]))
    state.pbest = p.copy()
    state.gbest = np.array([32.0, 32.0, 32.0])  # ignored, social weight is zero
    before = np.abs(state.position - p)
    for _ in range(30):
        state = step(state, space, cfg, rng)
        after = np.abs(state.position - p)
        assert np.all(after <= before + 1e-9)
        before = after
    assert np.allclose(state.position, p, atol=1e-3)


def test_velocity_clamping_is_respected():
    cfg = PSOConfig(v_max=2.0)
    rng = np.random.default_rng(1)
    space = SearchSpace(l_max=8)
    x = np.full((6, 8), 11.0)
    state = initial_state(x, np.arange(6.0))
    state.pbest = np.full((6, 8), 32.0)
    state.gbest = np.full(8, 32.0)
    state.velocity = np.full((6, 8), 50.0)
    for _ in range(20):
        state = step(state, space, cfg, rng)
        assert np.abs(state.velocity).max() <= 2.0 + 1e-12


def test_hitting_a_bound_zeroes_that_velocity_component():
    """The absorbing wall. Not in the paper; without it a swarm parks on the
    bound with the velocity still pushing outward."""
    cfg = PSOConfig()
    space = SearchSpace(l_max=2)
    x = np.array([[11.5, 31.0]])
    v = np.array([[-4.0, 5.0]])
    new_x, new_v = position_update(x, v, space, cfg)
    assert new_x.tolist() == [[11.0, 32.0]]
    assert new_v.tolist() == [[0.0, 0.0]]


def test_positions_stay_inside_the_encodable_range():
    cfg = PSOConfig()
    rng = np.random.default_rng(2)
    x = rng.uniform(SPACE.lower_bound, SPACE.upper_bound, (10, SPACE.l_max))
    state = initial_state(x, rng.random(10))
    state.velocity = rng.normal(0, 40.0, x.shape)
    for _ in range(50):
        state = step(state, SPACE, cfg, rng)
        assert state.position.min() >= SPACE.lower_bound - 1e-12
        assert state.position.max() <= SPACE.upper_bound + 1e-12
        for p in state.position:
            assert len(decode(p, SPACE)) <= SPACE.l_max


def test_personal_best_only_moves_on_a_strict_improvement():
    """Sec. 4.5, point 1, and Alg. 3 lines 11-13."""
    x = np.array([[12.0], [12.0]])
    state = initial_state(x, np.array([1.0, 1.0]))
    state.position = np.array([[20.0], [20.0]])
    state = update_bests(state, np.array([0.5, 1.0]))
    assert state.pbest[0].tolist() == [20.0]  # improved
    assert state.pbest[1].tolist() == [12.0]  # tie, keeps the old one
    assert state.gbest_fitness == 0.5


def test_global_best_is_the_best_personal_best():
    """Eq. 3: the min over particles and over all iterations so far."""
    rng = np.random.default_rng(3)
    x = rng.uniform(11, 32, (7, 4))
    state = initial_state(x, np.array([5.0, 3.0, 9.0, 4.0, 8.0, 7.0, 6.0]))
    assert state.gbest_fitness == 3.0
    state.position = rng.uniform(11, 32, (7, 4))
    state = update_bests(state, np.array([9.0, 9.0, 9.0, 2.0, 9.0, 9.0, 9.0]))
    assert state.gbest_fitness == 2.0
    assert np.allclose(state.gbest, state.position[3])
    # a generation where nobody improves cannot make the global best worse
    state = update_bests(state, np.full(7, 100.0))
    assert state.gbest_fitness == 2.0


def test_swarm_converges_on_a_problem_with_a_known_optimum():
    """Sphere centred on a fixed point well inside the encodable box.

    Kept away from the bounds on purpose. A target sitting one unit from a wall
    is a separate and much harder problem for PSO with clamped positions, and
    it is not what this test is about.
    """
    space = SearchSpace(l_max=8)
    target = np.array([14.0, 18.0, 25.0, 29.0, 15.0, 22.0, 28.0, 16.0])

    def objective(x):
        return ((x - target) ** 2).sum(axis=1)

    for seed in range(5):
        rng = np.random.default_rng(seed)
        state, history = minimize(objective, space, 25, 250, rng)
        assert history[-1] <= history[0]
        assert np.max(np.abs(state.gbest - target)) < 0.5
        assert np.all(np.diff(history) <= 1e-12)  # gbest never gets worse


def test_swarm_finds_the_optimum_of_a_discrete_encoding_objective():
    """Same thing but scored on the decoded architecture, so the objective is
    a step function of position, which is what Alg. 3 actually optimizes."""
    space = SearchSpace(l_max=6, r_lower=12, r_upper=32)
    wanted = [32, 12, 24]

    def objective(x):
        out = []
        for row in x:
            rates = decode(row, space)
            pad = rates[: len(wanted)] + [0] * (len(wanted) - len(rates))
            out.append(
                abs(len(rates) - len(wanted)) * 20.0
                + sum(abs(a - b) for a, b in zip(pad, wanted))
            )
        return np.asarray(out, dtype=float)

    rng = np.random.default_rng(0)
    state, history = minimize(objective, space, 30, 150, rng)
    assert history[-1] == 0.0
    assert decode(state.gbest, space) == wanted


def test_a_swarm_with_no_randomness_and_no_social_pull_stands_still():
    """Sanity check on the fixed points of Eq. 4: a particle already at its
    personal best with zero velocity does not move."""
    cfg = PSOConfig(inertia=0.0)
    space = SearchSpace(l_max=4)
    x = np.array([[20.0, 20.0, 20.0, 20.0]])
    state = initial_state(x, np.array([1.0]))
    rng = np.random.default_rng(0)
    for _ in range(10):
        state = step(state, space, cfg, rng)
    assert np.allclose(state.position, x)


def test_a_particle_can_change_block_length_during_a_run():
    """Variable length is reached through the flag, not through ragged
    vectors (Sec. 4.5). A coordinate crossing r_lower toggles a layer."""
    cfg = PSOConfig()
    rng = np.random.default_rng(7)
    space = SearchSpace(l_max=6)
    x = np.full((4, 6), 11.0)  # every layer off
    state = initial_state(x, np.arange(4.0))
    state.pbest = rng.uniform(11.0, 32.0, (4, 6))
    state.gbest = np.array([30.0, 11.0, 28.0, 11.0, 26.0, 11.0])
    lengths = {0}
    for _ in range(20):
        state = step(state, space, cfg, rng)
        lengths.update(len(decode(p, space)) for p in state.position)
    assert len(lengths) > 3
    assert max(lengths) > 0


def test_a_shorter_particle_is_pulled_toward_a_longer_personal_best():
    """Length mismatch needs no special case: the cognitive term acts on the
    flag cells like any other coordinate."""
    cfg = PSOConfig(inertia=0.0, social=0.0, cognitive=1.5)
    rng = np.random.default_rng(8)
    space = SearchSpace(l_max=5)
    short = np.array([[20.0, 11.0, 11.0, 11.0, 11.0]])
    long = np.array([[20.0, 22.0, 24.0, 26.0, 28.0]])
    state = initial_state(short, np.array([1.0]))
    state.pbest = long.copy()
    state.gbest = long[0].copy()
    for _ in range(25):
        state = step(state, space, cfg, rng)
    assert decode(state.position[0], space) == decode(long[0], space)


def test_the_swarm_beats_uniform_sampling_on_an_equal_budget():
    """The claim the toy end-to-end search makes, but on a cheap deterministic
    objective so it can be run over many seeds instead of one.

    Rosenbrock, mapped onto the encoded box, with the swarm and the sampler
    each allowed the same number of objective calls.
    """
    space = SearchSpace(l_max=8)

    def objective(x):
        z = (x - space.lower_bound) / (space.upper_bound - space.lower_bound) * 4.0 - 2.0
        return (
            100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1.0 - z[:, :-1]) ** 2
        ).sum(axis=1)

    n_particles, n_generations = 10, 40
    budget = n_particles * (n_generations + 1)
    wins, ratios = 0, []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        state, _ = minimize(objective, space, n_particles, n_generations, rng)
        sampler = np.random.default_rng(seed)
        drawn = sampler.uniform(
            space.lower_bound, space.upper_bound, (budget, space.l_max)
        )
        best_sampled = float(objective(drawn).min())
        wins += state.gbest_fitness < best_sampled
        ratios.append(state.gbest_fitness / best_sampled)
    assert wins == 20
    assert float(np.median(ratios)) < 0.5
