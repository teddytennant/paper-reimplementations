import numpy as np

from tauleap_schedule.eps_fact import eps_fact
from tauleap_schedule.rho import bernstein_rho, binomial_mixture_rho
from tauleap_schedule.schedules import uniform


def test_iota_zero_implies_rho_zero_and_eps_fact_zero():
    N = 6
    iota = np.zeros(N - 1)
    u = np.linspace(0.0, 1.0, 25)
    rho_u = np.asarray(bernstein_rho(u, iota, N))
    assert np.max(np.abs(rho_u)) < 1e-10

    def rho_fn(x):
        return bernstein_rho(x, iota, N)

    beta = uniform(5)
    assert abs(eps_fact(beta, rho_fn, N)) < 1e-8


def test_bernstein_matches_binomial_mixture_on_random_iota():
    rng = np.random.default_rng(0)
    N = 7
    iota = rng.random(N - 1)
    u = np.linspace(0.0, 1.0, 21)
    r_b = np.asarray(bernstein_rho(u, iota, N), dtype=np.float64)
    r_m = np.asarray(binomial_mixture_rho(u, iota, N), dtype=np.float64)
    np.testing.assert_allclose(r_b, r_m, rtol=1e-5, atol=1e-6)
