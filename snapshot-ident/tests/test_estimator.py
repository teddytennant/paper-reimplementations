"""Cor. 18 curl-blind baseline and weak-form recovery of planted rotation."""

import jax
import jax.numpy as jnp

from snapshot_ident.estimator import (
    antisym,
    gaussian_ot_map,
    gradient_only_fit,
    mean_displacement_field,
    rel_err_solenoidal,
    ridge_fit,
)
from snapshot_ident.moments import weak_design_from_samples, weak_design_from_second_moments
from snapshot_ident.sde import (
    planted_curl_H,
    rotation_generator,
    sample_independent_slices,
    snapshot_covariances,
)


D = 4
M = 2
NU = 0.5
SIGMA2 = 0.5
OMEGA = 0.8
TIMES = jnp.array([0.0, 0.5, 1.0, 1.5], dtype=jnp.float64)
SIGMA0_ANISO = jnp.diag(jnp.array([2.8, 0.22, 1.3, 0.7], dtype=jnp.float64))
SIGMA0_ISO = 1.2 * jnp.eye(D, dtype=jnp.float64)


def _setup(Sigma0, omega=OMEGA):
    H_star = planted_curl_H(D, M, NU, omega)
    Omega = rotation_generator(D, M, omega)
    Sigmas = snapshot_covariances(Sigma0, TIMES, NU, SIGMA2, Omega)
    dts = jnp.diff(TIMES)
    return H_star, Sigmas, dts


def test_displacement_velocity_has_zero_antisymmetric_part():
    """Cor. 18: OT / displacement interpolant of centered Gaussians is a gradient."""
    _, Sigmas, dts = _setup(SIGMA0_ANISO)
    for k in range(dts.shape[0]):
        A = gaussian_ot_map(Sigmas[k], Sigmas[k + 1])
        assert jnp.allclose(A, A.T, rtol=1e-10, atol=1e-12)
        assert jnp.linalg.norm(antisym(A)) < 1e-10
    V = mean_displacement_field(Sigmas, dts)
    assert jnp.linalg.norm(antisym(V)) < 1e-10


def test_gradient_only_relerr_is_one_on_isotropic_rotation():
    """Matched isotropic marginals: gradient-only / displacement has relErr = 1."""
    H_star, Sigmas, dts = _setup(SIGMA0_ISO, omega=OMEGA)
    X, bb = weak_design_from_second_moments(Sigmas, dts, SIGMA2)
    H_grad = gradient_only_fit(X, bb, D, zeta=0.0)
    assert jnp.linalg.norm(antisym(H_grad)) < 1e-10
    rel = rel_err_solenoidal(H_grad, H_star)
    assert jnp.allclose(rel, 1.0, rtol=0.0, atol=1e-10)

    V = mean_displacement_field(Sigmas, dts)
    assert jnp.linalg.norm(antisym(V)) < 1e-10
    assert jnp.allclose(rel_err_solenoidal(V, H_star), 1.0, atol=1e-10)


def test_weak_form_recovers_planted_curl():
    """Anisotropic covariances: weak-form solenoidal relErr is strictly below 1."""
    key = jax.random.PRNGKey(0)
    H_star, Sigmas, dts = _setup(SIGMA0_ANISO, omega=OMEGA)

    X, bb = weak_design_from_second_moments(Sigmas, dts, SIGMA2)
    H_weak = ridge_fit(X, bb, D, zeta=1e-10, restrict="full")
    H_grad = gradient_only_fit(X, bb, D, zeta=1e-10)
    rel_weak = float(rel_err_solenoidal(H_weak, H_star))
    rel_grad = float(rel_err_solenoidal(H_grad, H_star))
    assert rel_grad == 1.0 or abs(rel_grad - 1.0) < 1e-8
    assert rel_weak < 1.0
    assert rel_weak < 0.5

    slices = sample_independent_slices(key, n=8000, Sigmas=Sigmas)
    Xs, bbs = weak_design_from_samples(slices, dts, SIGMA2)
    H_s = ridge_fit(Xs, bbs, D, zeta=1e-6, restrict="full")
    rel_s = float(rel_err_solenoidal(H_s, H_star))
    assert rel_s < 1.0
