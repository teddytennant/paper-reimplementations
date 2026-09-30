"""Gauge diagnostics: Prop. 22 isotropic rotation and Prop. 15 curvature."""

import jax.numpy as jnp

from snapshot_ident.estimator import prop15_schur_eigs, weak_solenoidal_gram_eigs
from snapshot_ident.moments import weak_design_from_second_moments
from snapshot_ident.sde import rotation_generator, snapshot_covariances


D = 4
M = 2
NU = 0.5
SIGMA2 = 0.5
OMEGA = 0.9
TIMES = jnp.array([0.0, 0.5, 1.0, 1.5], dtype=jnp.float64)


def test_isotropic_block_rotation_leaves_marginals_unchanged():
    """Prop. 22: isotropic-block rotation does not change centered Gaussian slices."""
    Sigma0 = 1.3 * jnp.eye(D, dtype=jnp.float64)
    Omega = rotation_generator(D, M, OMEGA)
    Omega0 = jnp.zeros((D, D), dtype=jnp.float64)
    with_rot = snapshot_covariances(Sigma0, TIMES, NU, SIGMA2, Omega)
    no_rot = snapshot_covariances(Sigma0, TIMES, NU, SIGMA2, Omega0)
    assert jnp.allclose(with_rot, no_rot, rtol=1e-10, atol=1e-12)
    for k in range(TIMES.shape[0]):
        # Each slice remains a multiple of I, independent of Omega.
        S = with_rot[k]
        off = S - jnp.diag(jnp.diag(S))
        assert jnp.max(jnp.abs(off)) < 1e-10
        assert jnp.allclose(jnp.diag(S), S[0, 0] * jnp.ones(D), rtol=1e-10, atol=1e-12)


def test_prop15_isotropic_kernel_vs_anisotropic_visible():
    """Prop. 15: solenoidal Schur is a kernel when all Σ_k ∝ I, and larger otherwise."""
    Omega = rotation_generator(D, M, OMEGA)
    Sigma_iso = 1.1 * jnp.eye(D, dtype=jnp.float64)
    iso = snapshot_covariances(Sigma_iso, TIMES, NU, SIGMA2, Omega)
    eigs_iso = prop15_schur_eigs(iso)

    Sigma_aniso = jnp.diag(jnp.array([2.6, 0.25, 1.4, 0.55], dtype=jnp.float64))
    aniso = snapshot_covariances(Sigma_aniso, TIMES, NU, SIGMA2, Omega)
    eigs_aniso = prop15_schur_eigs(aniso)

    assert float(eigs_iso[0]) < 1e-8
    assert float(eigs_aniso[0]) > 1e-2
    assert float(eigs_aniso[0]) > 50.0 * float(jnp.abs(eigs_iso[0]) + 1e-16)

    dts = jnp.diff(TIMES)
    X_iso, _ = weak_design_from_second_moments(iso, dts, SIGMA2)
    X_aniso, _ = weak_design_from_second_moments(aniso, dts, SIGMA2)
    g_iso = weak_solenoidal_gram_eigs(X_iso, D)
    g_aniso = weak_solenoidal_gram_eigs(X_aniso, D)
    assert float(g_iso[0]) < 1e-10
    assert float(g_aniso[0]) > 1e-4
