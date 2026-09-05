"""The EMA cascade against the filter algebra it is supposed to implement.

Everything here runs in float64 (echo_oftrl turns x64 on at import) and the
tolerances are set accordingly: the recursion and the reference filters are the
same linear operator computed two different ways, so agreement to 1e-11 is what
float64 gives, not a fudge.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from echo_oftrl import apply_A, apply_delta, apply_I_minus_delta, lag, run_cascade
from echo_oftrl.cascade import ema_cascade_step

RHOS = [0.0, 0.5, 2.0 / 3.0, 0.9]


def _random_sequence(t=120, m=4, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(t, m)))


@pytest.mark.parametrize("rho", RHOS[1:])
def test_filter_identity_one_pole_ema(rho):
    """I - delta = (1 - rho) L (I - rho L)^-1, the identity in Section 1.1."""
    z = _random_sequence(seed=1)
    lhs = apply_I_minus_delta(z, rho)
    rhs = (1.0 - rho) * apply_A(lag(z), rho)
    assert np.allclose(lhs, rhs, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rho", RHOS[1:])
def test_i_minus_delta_averages_strictly_past_values(rho):
    """((I-delta) z)^(t) = (1-rho) sum_{k>=1} rho^(k-1) z^(t-k), Section 3.3."""
    z = np.asarray(_random_sequence(seed=2))
    t_max, m = z.shape
    direct = np.zeros_like(z)
    for t in range(t_max):  # index t here is round t+1 in the paper
        for k in range(1, t + 1):
            direct[t] += (1.0 - rho) * rho ** (k - 1) * z[t - k]
    assert np.allclose(apply_I_minus_delta(jnp.asarray(z), rho), direct, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rho,depth", [(0.5, 2), (2.0 / 3.0, 3), (0.75, 4), (0.9, 6)])
def test_cascade_state_matches_closed_form(rho, depth):
    """ema_h^(t) = ((I - delta) delta^(h-1) g)^(t), the induction after (3.6)."""
    g = _random_sequence(seed=3)
    emas, _ = run_cascade(g, rho, depth)
    filtered = g
    for h in range(depth):
        expected = apply_I_minus_delta(filtered, rho)
        assert np.allclose(emas[:, h], expected, rtol=0, atol=1e-11)
        filtered = apply_delta(filtered, rho)


@pytest.mark.parametrize("rho,depth", [(0.5, 2), (2.0 / 3.0, 3), (0.9, 6)])
def test_predictor_and_error_are_the_nth_order_filter(rho, depth):
    """ghat = (I - delta^H) g and g - ghat = delta^H g, equation (3.3)."""
    g = _random_sequence(seed=4)
    _, ghats = run_cascade(g, rho, depth)
    residual = g
    for _ in range(depth):
        residual = apply_delta(residual, rho)
    assert np.allclose(ghats, g - residual, rtol=0, atol=1e-11)


def test_depth_one_rho_zero_is_last_gradient_optimism():
    """The collapse used as the comparison baseline: ghat^(t) = g^(t-1)."""
    g = _random_sequence(seed=5)
    _, ghats = run_cascade(g, rho=0.0, depth=1)
    assert np.array_equal(np.asarray(ghats), np.asarray(lag(g)))


def test_rho_one_switches_optimism_off():
    """depth 1, rho 1 leaves the state at zero, so ghat = 0 and OFTRL is FTRL."""
    g = _random_sequence(seed=6)
    _, ghats = run_cascade(g, rho=1.0, depth=1)
    assert np.array_equal(np.asarray(ghats), np.zeros_like(np.asarray(g)))


def test_cascade_step_error_equals_g_minus_predictor():
    """Algorithm 1 line 8 telescopes to e_H = g - sum_h ema_h."""
    rng = np.random.default_rng(7)
    ema = jnp.asarray(rng.normal(size=(5, 3)))
    g = jnp.asarray(rng.normal(size=(3,)))
    _, err = ema_cascade_step(ema, g, rho=0.8)
    assert np.allclose(err, g - jnp.sum(ema, axis=0), rtol=0, atol=1e-14)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8, 12])
def test_preconditioning_caps_the_frequency_gain(n):
    """Section 1.1's reason for delta over the raw difference.

    The N-fold raw difference (I - L)^N has worst-case gain 2^N on the unit
    circle. With rho = 1 - 1/N the paper claims sup |delta(w)|^N = (2/(1+rho))^N,
    which is under sqrt(e) = 1.649 for every N and under the paper's stated e.
    """
    rho = 1.0 - 1.0 / n
    w = np.linspace(0.0, 2.0 * np.pi, 20001)
    z = np.exp(-1j * w)
    gain = np.abs((1.0 - z) / (1.0 - rho * z))
    assert gain.max() == pytest.approx(2.0 / (1.0 + rho), rel=1e-6)
    assert gain.max() ** n <= np.e
    assert np.abs(1.0 - z).max() ** n == pytest.approx(2.0**n, rel=1e-6)
