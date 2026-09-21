import numpy as np

from tauleap_schedule.eps_fact import eps_fact
from tauleap_schedule.schedules import uniform


def _const(c):
    def rho(u):
        u = np.asarray(u, dtype=np.float64)
        return np.full_like(u, c, dtype=np.float64)

    return rho


def test_constant_rho_uniform_schedule_closed_form():
    N = 10
    c = 1.7
    K = 8
    beta = uniform(K)
    got = eps_fact(beta, _const(c), N)
    want = N * c / (2.0 * K)
    assert abs(got - want) < 1e-6
