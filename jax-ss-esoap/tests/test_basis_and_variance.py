"""The adaptive basis trigger, equation (4), and the variance-state transition
of Section 3.2 with the bound from Theorem 4.1."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import quadratic as qd
from ss_esoap import gamma_from_rho, offdiag_ratio, ss_esoap
from ss_esoap._core import Config, eigh_basis, init_leaf, update_leaf
from ss_esoap.optimizer import scale_by_ss_esoap


def _spd(rng, d):
    m = rng.standard_normal((d, d))
    return jnp.asarray(m @ m.T / d + np.eye(d))


# ---------------------------------------------------------------- equation (4)


def test_offdiag_ratio_is_zero_in_the_exact_eigenbasis():
    a = _spd(np.random.default_rng(0), 12)
    assert float(offdiag_ratio(a, eigh_basis(a), 0.0)) < 1e-12


def test_offdiag_ratio_is_positive_in_a_stale_basis():
    rng = np.random.default_rng(1)
    a = _spd(rng, 12)
    q = eigh_basis(_spd(rng, 12))
    assert float(offdiag_ratio(a, q, 0.0)) > 0.1


def test_offdiag_ratio_is_scale_invariant():
    """It is a ratio of norms of the same matrix, so a rescaling of the tracked
    Kronecker factor must not move the trigger."""
    a = _spd(np.random.default_rng(2), 10)
    q = jnp.eye(10)
    assert jnp.allclose(
        offdiag_ratio(a, q, 0.0), offdiag_ratio(1024.0 * a, q, 0.0), rtol=1e-12
    )


@pytest.mark.parametrize(
    "rho,expected", [(0.95, 0.25), (0.81, 0.25), (0.79, 0.5), (0.51, 0.5),
                     (0.49, 0.75), (0.21, 0.75)]
)
def test_gamma_ladder(rho, expected):
    """Section 3.2 / Table 6: 0.25 above 0.8, 0.5 above 0.5, 0.75 otherwise."""
    assert float(gamma_from_rho(jnp.asarray(rho))) == expected


# ---------------------------------------------------------------- the trigger


def _count_basis_refreshes(cfg, steps=200, seed=0, lr=1e-2):
    a, b, c, x0, _ = qd.make(seed)
    tx = optax.chain(scale_by_ss_esoap(cfg), optax.scale(-lr))
    state = tx.init(x0)

    @jax.jit
    def step(x, state):
        g = jax.grad(qd.loss)(x, a, b, c)
        u, state = tx.update(g, state, x)
        return optax.apply_updates(x, u), state

    x, fires = x0, 0
    for _ in range(steps):
        before = state[0].leaves.ql
        x, state = step(x, state)
        if not bool(jnp.array_equal(before, state[0].leaves.ql)):
            fires += 1
    return fires


def test_trigger_is_selective_and_monotone_in_the_threshold():
    """Table 11's picture: a threshold of zero refreshes every step, a large one
    barely refreshes, and the default sits in between."""
    steps = 120
    always = _count_basis_refreshes(Config(trigger=0.0), steps)
    default = _count_basis_refreshes(Config(trigger=0.2), steps)
    lazy = _count_basis_refreshes(Config(trigger=0.95), steps)
    assert always >= steps - 1
    assert 0 < default < steps
    assert lazy < default


def test_refresh_drops_the_off_diagonal_mass_below_the_threshold():
    """After a refresh the new basis diagonalizes the factor it was built from,
    so rho falls to zero at that instant."""
    cfg = Config(trigger=0.2)
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal((8, 6)))
    state = init_leaf(p, cfg.eps, cfg.max_precond_dim)
    g = jnp.asarray(rng.standard_normal((8, 6)))
    _, state = update_leaf(g, state, p, jnp.asarray(5), cfg)
    rho = max(
        float(offdiag_ratio(state.left, state.ql, cfg.eps)),
        float(offdiag_ratio(state.right, state.qr, cfg.eps)),
    )
    assert rho < 1e-10


# -------------------------------------------------- variance-state transition


def _warm_state(steps=6, shape=(10, 7), seed=7):
    """A state with a nonzero second moment and a deliberately stale basis.

    Warmup runs with basis updates off, so the basis is still the identity and
    the refresh at the next step is a real rotation. All the transition variants
    then branch from this one state, which is what makes the comparison below a
    controlled one.
    """
    cfg = Config(basis_updates=False, self_scale=False)
    rng = np.random.default_rng(seed)
    p = jnp.asarray(rng.standard_normal(shape))
    state = init_leaf(p, cfg.eps, cfg.max_precond_dim)
    for _ in range(steps):
        g = jnp.asarray(rng.standard_normal(shape))
        _, state = update_leaf(g, state, p, jnp.asarray(1), cfg)
    return p, state, jnp.asarray(rng.standard_normal(shape))


def _one_forced_refresh(variance_transition, gamma, warm=None):
    """Take one step, from a shared warm state, whose basis refresh is forced."""
    cfg = Config(
        trigger=0.0, variance_transition=variance_transition, gamma=gamma,
        self_scale=False,
    )
    p, state, g = _warm_state() if warm is None else warm
    u, new = update_leaf(g, state, p, jnp.asarray(9), cfg)
    return u, new, state.nu


@pytest.mark.parametrize("gamma", [0.25, 0.5, 0.75])
def test_downscaling_keeps_the_coordinatewise_shape(gamma):
    """Section 3.2: V keeps its shape and is multiplied by gamma, rather than
    being carried through the similarity transform."""
    cfg = Config(trigger=0.0, gamma=gamma, self_scale=False, b2=0.95)
    rng = np.random.default_rng(3)
    p = jnp.asarray(rng.standard_normal((9, 6)))
    state = init_leaf(p, cfg.eps, cfg.max_precond_dim)
    for _ in range(5):
        _, state = update_leaf(
            jnp.asarray(rng.standard_normal((9, 6))), state, p,
            jnp.asarray(1), cfg,
        )
    nu_before, g = state.nu, jnp.asarray(rng.standard_normal((9, 6)))
    _, new = update_leaf(g, state, p, jnp.asarray(9), cfg)
    gt = new.ql.T @ (g @ new.qr)
    expected = cfg.b2 * gamma * nu_before + (1.0 - cfg.b2) * gt**2
    assert jnp.allclose(new.nu, expected, rtol=1e-12)


@pytest.mark.parametrize("gamma", [0.25, 0.5, 0.75])
def test_step_magnitude_is_continuous_across_the_rotation(gamma):
    """The point of downscaling rather than resetting.

    Everything except V is identical between the two runs, and the update in
    parameter space is an orthogonal rotation of the eigenspace update, so the
    Frobenius norm carries over. Since ``gamma V <= V_gamma_new`` and
    ``V_gamma_new <= V_1_new`` hold elementwise, the step norm is pinned to

        ||U_1||  <=  ||U_gamma||  <=  gamma^{-1/2} ||U_1||,

    a bounded, continuous change rather than a jump.
    """
    warm = _warm_state()
    u_keep, _, _ = _one_forced_refresh("downscale", 1.0, warm)
    u_gamma, _, _ = _one_forced_refresh("downscale", gamma, warm)
    lo, hi = float(jnp.linalg.norm(u_keep)), float(jnp.linalg.norm(u_gamma))
    assert lo <= hi + 1e-12
    assert hi <= lo / np.sqrt(gamma) + 1e-12


def test_hard_reset_jumps_where_downscaling_does_not():
    """A hard reset of V is the alternative Appendix D.1 compares against; it
    blows the step magnitude well past the downscaling bound."""
    warm = _warm_state()
    u_keep, _, _ = _one_forced_refresh("downscale", 1.0, warm)
    u_min, _, _ = _one_forced_refresh("downscale", 0.25, warm)
    u_reset, _, _ = _one_forced_refresh("reset", None, warm)
    ratio_down = float(jnp.linalg.norm(u_min) / jnp.linalg.norm(u_keep))
    ratio_reset = float(jnp.linalg.norm(u_reset) / jnp.linalg.norm(u_keep))
    assert ratio_down <= 2.0  # 0.25^{-1/2}
    assert ratio_reset > 3.0 * ratio_down


@pytest.mark.parametrize("gamma", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_theorem_4_1_mismatch_bound(gamma, seed):
    """Theorem 4.1, checked numerically on the objects it is stated over.

        ||gamma V_t - V*_{t+1}||_F
            <= (1 - gamma)||V_t||_F + 2||C_t||_2 ||I - U||_F + ||C_{t+1} - C_t||_F

    with ``V = D_Q(C) = diag(Q^T C Q)`` and ``U = Q_{t+1}^T Q_t``. The theorem's
    Frobenius norms are of diagonal matrices, so they are 2-norms of the
    diagonals.
    """
    rng = np.random.default_rng(seed)
    d = 14
    c_t = _spd(rng, d)
    drift = 0.15 * _spd(rng, d)
    c_next = c_t + drift
    q_t = eigh_basis(c_t)
    q_next = eigh_basis(c_next)

    v_t = jnp.diagonal(q_t.T @ c_t @ q_t)
    v_star = jnp.diagonal(q_next.T @ c_next @ q_next)
    u = q_next.T @ q_t

    lhs = jnp.linalg.norm(gamma * v_t - v_star)
    rhs = (
        (1.0 - gamma) * jnp.linalg.norm(v_t)
        + 2.0 * jnp.linalg.norm(c_t, 2) * jnp.linalg.norm(jnp.eye(d) - u)
        + jnp.linalg.norm(c_next - c_t)
    )
    assert float(lhs) <= float(rhs) + 1e-9


def test_reprojection_can_drive_the_second_moment_negative():
    """Why the SOAP baseline needs a clamp, and part of why Section 2.4 argues
    against reprojection: the similarity transform of a matrix-shaped V is not
    positivity preserving."""
    rng = np.random.default_rng(11)
    v = jnp.asarray(rng.random((8, 6)))
    q_old_l, q_new_l = eigh_basis(_spd(rng, 8)), eigh_basis(_spd(rng, 8))
    q_old_r, q_new_r = eigh_basis(_spd(rng, 6)), eigh_basis(_spd(rng, 6))
    rotated = (q_new_l.T @ q_old_l) @ v @ (q_old_r.T @ q_new_r)
    assert float(jnp.min(rotated)) < 0.0


def test_optimizer_runs_with_every_variance_transition():
    a, b, c, x0, _ = qd.make(0)
    for mode in ("downscale", "reproject", "reset", "keep"):
        opt = ss_esoap(1e-2, variance_transition=mode)
        state = opt.init(x0)

        @jax.jit
        def step(x, state):
            g = jax.grad(qd.loss)(x, a, b, c)
            u, state = opt.update(g, state, x)
            return optax.apply_updates(x, u), state

        x = x0
        for _ in range(20):
            x, state = step(x, state)
        assert jnp.all(jnp.isfinite(x)), mode
