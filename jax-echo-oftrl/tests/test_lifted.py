"""The lifted simplex, the regularizer of Section 3.2, and Lemma 3.1's map Q_m."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from echo_oftrl import gamma_const, lifted_argmax, psi_tilde, theoretical_learning_rate, xi_const
from echo_oftrl.lifted import stationarity

SIZES = [2, 3, 5, 8]
SCALES = [0.1, 1.0, 10.0, 100.0]


def _thetas(m, seed=0):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.normal(size=m) * s) for s in SCALES]


@pytest.mark.parametrize("m", SIZES)
def test_lifted_argmax_zeroes_the_stationarity_function(m):
    """lam is the root of the strictly increasing function in Lemma 3.1."""
    gamma = gamma_const(m)
    for theta in _thetas(m):
        lam, _ = lifted_argmax(theta, gamma)
        assert 0.0 < float(lam) < 1.0
        assert abs(float(stationarity(lam, theta, gamma))) < 1e-8


@pytest.mark.parametrize("m", SIZES)
def test_stationarity_is_strictly_increasing(m):
    """The monotonicity Lemma 3.1 asserts, which is what makes bisection legal."""
    gamma = gamma_const(m)
    grid = jnp.asarray(np.linspace(1e-6, 1.0 - 1e-9, 4000))
    for theta in _thetas(m, seed=1):
        vals = np.asarray(jax.vmap(lambda l: stationarity(l, theta, gamma))(grid))
        assert np.all(np.diff(vals) > 0.0)
        assert vals[0] < 0.0 < vals[-1]


def _objective_numpy(theta, lam, x, gamma):
    """<theta, y> - psitilde_m(lam, y) at y = lam x, written out in numpy.

    A second, independent transcription of the Section 3.2 definitions, so this
    is not comparing echo_oftrl.lifted against itself.
    """
    theta = np.asarray(theta, dtype=np.float64)
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    safe = np.clip(x, 1e-300, None)
    psi = np.sum(np.where(x > 0, x * np.log(safe), 0.0), axis=-1)
    reg = np.where(
        lam > 0,
        -np.sqrt(np.maximum(1.0 - lam, 0.0)) + np.sqrt(lam) * (psi - 2.0 * gamma - 3.0),
        -1.0,
    )
    return lam * (x @ theta) - reg


@pytest.mark.parametrize("m", SIZES)
def test_lifted_argmax_is_the_maximizer(m):
    """Q_m(theta) maximizes <theta, y> - psitilde_m(lam, y) over the lifted simplex.

    Checked against 20000 random feasible points and against perturbations of
    the returned point in lam and in x. The slack is 1e-9 rather than 0 because
    the closest perturbations sit 1e-5 from the optimum, where the objective is
    flat to second order.
    """
    gamma = gamma_const(m)
    rng = np.random.default_rng(2)
    for theta in _thetas(m, seed=2):
        lam, x = lifted_argmax(theta, gamma)
        lam, x = float(lam), np.asarray(x)
        best = float(_objective_numpy(theta, lam, x, gamma)[0])

        cand_lam = rng.uniform(1e-9, 1.0 - 1e-12, 20000)
        conc = rng.choice([0.1, 1.0, 10.0], size=20000)[:, None]
        cand_x = rng.gamma(conc * np.ones((20000, m)))
        cand_x /= cand_x.sum(axis=1, keepdims=True)
        assert _objective_numpy(theta, cand_lam, cand_x, gamma).max() <= best + 1e-9

        steps = np.array([1e-5, 1e-3, 1e-2, 0.1, -1e-5, -1e-3, -1e-2, -0.1])
        near_lam = np.clip(lam + steps, 1e-12, 1.0 - 1e-12)
        assert _objective_numpy(theta, near_lam, np.tile(x, (steps.size, 1)), gamma).max() <= best + 1e-9

        noise = rng.normal(size=(steps.size, m))
        shifted = np.log(np.clip(x, 1e-300, None)) + steps[:, None] * noise
        near_x = np.exp(shifted - shifted.max(axis=1, keepdims=True))
        near_x /= near_x.sum(axis=1, keepdims=True)
        assert _objective_numpy(theta, np.full(steps.size, lam), near_x, gamma).max() <= best + 1e-9


@pytest.mark.parametrize("m", SIZES)
def test_inner_problem_is_hedge_at_rate_sqrt_lambda(m):
    """x_m(theta) = softmax(sqrt(lam) theta), the display just above Section 3.4."""
    gamma = gamma_const(m)
    for theta in _thetas(m, seed=3):
        lam, x = lifted_argmax(theta, gamma)
        assert np.allclose(x, jax.nn.softmax(jnp.sqrt(lam) * theta), rtol=0, atol=1e-15)
        assert float(jnp.sum(x)) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("m", SIZES)
def test_gamma_bounds_the_log_probability_variance(m):
    """Section 3.2: Gamma_m bounds Var_{a~x}[-log x(a)] over the simplex."""
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(20000):
        x = rng.dirichlet(np.ones(m) * rng.choice([0.02, 0.1, 1.0, 10.0]))
        x = np.clip(x, 1e-300, None)
        logs = -np.log(x)
        worst = max(worst, float(np.sum(x * logs**2) - np.sum(x * logs) ** 2))
    assert worst <= gamma_const(m)


@pytest.mark.parametrize("m", SIZES)
def test_regularizer_range_matches_lemma_3_1(m):
    """sup psitilde - inf psitilde <= 2 Gamma_m + log m + 4, the last claim of Lemma 3.1."""
    gamma = gamma_const(m)
    rng = np.random.default_rng(5)
    n = 20000
    lams = np.concatenate([[0.0, 1.0, 1.0], rng.uniform(0.0, 1.0, n - 3)])
    xs = rng.gamma(rng.choice([0.02, 0.3, 3.0], size=n)[:, None] * np.ones((n, m)))
    xs /= xs.sum(axis=1, keepdims=True)
    xs[0] = xs[1] = 1.0 / m
    xs[2, 0] = 1.0 - 1e-12
    xs[2, 1:] = 1e-12 / (m - 1)
    vals = -_objective_numpy(np.zeros(m), lams, xs, gamma)
    assert float(vals.max() - vals.min()) <= 2.0 * gamma + math.log(m) + 4.0
    assert float(psi_tilde(jnp.asarray(0.0), jnp.zeros(m), gamma)) == pytest.approx(-1.0)


def test_theoretical_learning_rate_matches_theorem_3_3():
    """eta = 2^-94 / (N^20 Xi_max) with Xi_m = 10^4 (1 + Gamma_m)."""
    assert xi_const(4) == pytest.approx(1e4 * (1.0 + gamma_const(4)))
    assert theoretical_learning_rate(3, 4) == pytest.approx(2.0**-94 / (3**20 * xi_const(4)))
    assert theoretical_learning_rate(2, 2) < 1e-30
