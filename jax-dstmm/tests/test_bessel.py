"""log K_nu against scipy, and the derivative used by equation (22)."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln, kve

from dstmm import dlog_bessel_k_dnu, log_bessel_k


def test_log_bessel_k_matches_scipy():
    """The quadrature in dstmm.bessel must reproduce scipy's K_nu over the range
    the GHST density and Proposition 2 actually visit: order (nu+p)/2 and
    argument sqrt(q (nu + d^2))."""
    nus = np.array([0.5, 1.0, 1.5, 2.5, 4.5, 10.0, 25.0, 60.0, 101.0])
    xs = np.array([1e-6, 1e-4, 1e-2, 0.1, 1.0, 3.0, 10.0, 50.0, 200.0, 1000.0])
    order, arg = np.meshgrid(nus, xs, indexing="ij")
    ref = np.log(kve(order, arg)) - arg  # scipy overflows without the scaled form
    finite = np.isfinite(ref) & (kve(order, arg) > 0)
    assert finite.sum() > 60
    got = np.asarray(log_bessel_k(order, arg))
    assert np.abs(got - ref)[finite].max() < 1e-11


def test_log_bessel_k_small_argument_asymptote():
    """Where scipy overflows, check against K_nu(x) ~ Gamma(nu) (2/x)^nu / 2 as x -> 0."""
    nus = np.array([0.5, 2.5, 4.5, 10.0, 25.0, 60.0])
    x = 1e-8
    ref = gammaln(nus) - np.log(2.0) + nus * np.log(2.0 / x)
    got = np.asarray(log_bessel_k(nus, np.full_like(nus, x)))
    assert np.abs(got - ref).max() < 1e-6


def test_negative_order_folds():
    """K_{-nu} = K_nu, which is why Proposition 2's lambda = -(nu+p)/2 is usable."""
    x = np.array([0.4, 2.0, 30.0])
    assert np.allclose(log_bessel_k(-3.5, x), log_bessel_k(3.5, x), rtol=0, atol=1e-13)


@pytest.mark.parametrize("nu,x", [(2.5, 0.7), (4.5, 3.0), (-4.5, 3.0), (12.0, 40.0)])
def test_dlog_bessel_k_matches_finite_difference(nu, x):
    """Autodiff of the quadrature is the derivative equation (22) asks for."""
    h = 1e-5
    # abs() inside kve already carries the K_{-nu} = K_nu fold, so this is
    # d/dnu log K_nu with the sign the negative branch needs.
    fd = (np.log(kve(abs(nu + h), x)) - np.log(kve(abs(nu - h), x))) / (2 * h)
    got = float(dlog_bessel_k_dnu(nu, x))
    assert abs(got - fd) < 1e-7 * max(1.0, abs(fd))


def test_float64_is_on():
    """x64 has to be live or every likelihood in this package loses two digits."""
    assert jnp.zeros(1).dtype == jnp.float64
    assert log_bessel_k(2.5, 1.0).dtype == jnp.float64
