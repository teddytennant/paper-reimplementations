import numpy as np

from tauleap_schedule.functionals import D, DTC, TC


def _const(c):
    def rho(u):
        u = np.asarray(u, dtype=np.float64)
        return np.full_like(u, c, dtype=np.float64)

    return rho


def test_lemma2_constant_rho():
    c = 2.3
    N = 12
    rho = _const(c)
    assert abs(D(rho) - c) < 1e-6
    assert abs(TC(rho, N) - N * c / 2.0) < 1e-5
    assert abs(DTC(rho, N) - N * c / 2.0) < 1e-5
