import numpy as np

from tauleap_schedule.quad import integrate
from tauleap_schedule.schedules import (
    G_sqrt,
    geometric,
    limit_smooth,
    q_from_beta,
    reverse_geometric,
    shoot_finite_k,
    stationary_k2,
    uniform,
)


def _const(c):
    def rho(u):
        u = np.asarray(u, dtype=np.float64)
        return np.full_like(u, c, dtype=np.float64)

    return rho


def _rho_linear(u):
    u = np.asarray(u, dtype=np.float64)
    return 2.0 + u


def test_k2_stationarity_constant_and_linear_rho():
    x_const = stationary_k2(_const(3.0))
    assert abs(x_const - 0.5) < 1e-6

    x = stationary_k2(_rho_linear)
    lhs = integrate(_rho_linear, 0.0, x)
    rhs = (1.0 - x) * float(np.asarray(_rho_linear(np.array([x]))).reshape(-1)[0])
    assert abs(lhs - rhs) < 1e-4


def test_shooting_constant_rho_is_uniform():
    K = 8
    beta = shoot_finite_k(_const(1.0), K)
    expected = uniform(K)
    np.testing.assert_allclose(beta, expected, atol=1e-4)


def test_corollary2_constant_and_quadratic_g():
    t = np.linspace(0.0, 1.0, 11)
    beta_lin = limit_smooth(_const(4.0), t)
    np.testing.assert_allclose(beta_lin, t, atol=1e-5)

    def g(u):
        u = np.asarray(u, dtype=np.float64)
        return (1.0 + u) ** 2

    beta = limit_smooth(g, t)
    G_beta = G_sqrt(g, beta)
    G1 = float(G_sqrt(g, 1.0))
    np.testing.assert_allclose(G_beta, t * G1, atol=1e-4)


def test_geometric_increasing_ends_at_one_and_q_formula():
    K, a, N = 10, 0.4, 16
    beta = geometric(K, a, N)
    assert beta[0] == 0.0
    assert abs(beta[-1] - 1.0) < 1e-15
    assert np.all(np.diff(beta) >= -1e-15)

    q = q_from_beta(beta)
    for k in range(1, beta.size):
        denom = 1.0 - beta[k - 1]
        if denom > 0:
            want = (beta[k] - beta[k - 1]) / denom
        else:
            want = 1.0
        assert abs(q[k - 1] - want) < 1e-12
    assert abs(q[-1] - 1.0) < 1e-12

    rev = reverse_geometric(K, a, N)
    assert rev[0] == 0.0 and abs(rev[-1] - 1.0) < 1e-15
    assert np.all(np.diff(rev) >= -1e-15)
