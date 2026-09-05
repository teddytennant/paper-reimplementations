"""Equations (1) and (2): the discriminative subspace and the active dimension."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from caster import Caster, CasterConfig  # noqa: E402
from caster.subspace import active_dim, lda_directions, scatter_matrices, subspace_dim  # noqa: E402
from caster.synthetic import class_parameters, sample_classes  # noqa: E402


@pytest.mark.parametrize(
    "num_classes,k_max,dim,expected",
    [(10, 64, 32, 9), (10, 4, 32, 4), (100, 64, 32, 32), (3, 64, 32, 2)],
)
def test_subspace_dim_follows_eq1(num_classes, k_max, dim, expected):
    """k0 = min(k_max, C - 1), and never more than the feature dimension."""
    assert subspace_dim(num_classes, k_max, dim) == expected


@pytest.mark.parametrize(
    "k0,batch_size,expected", [(9, 100, 9), (9, 6, 5), (9, 10, 9), (9, 2, 1)]
)
def test_active_dim_follows_eq2(k0, batch_size, expected):
    """k = min(k0, B - 1)."""
    assert active_dim(k0, batch_size) == expected


def test_adapt_uses_the_batch_limited_subspace():
    """A batch smaller than k0 + 1 must shrink every array to k = B - 1."""
    key = jax.random.PRNGKey(0)
    num_classes, dim = 12, 20
    params = class_parameters(key, num_classes, dim, separation=3.0)
    features, labels = sample_classes(jax.random.PRNGKey(1), params, 40)
    model = Caster.fit(features, labels, num_classes, CasterConfig())
    assert model.state.k0 == 11

    result = model.adapt(features[:6])
    assert result.k == 5
    assert result.z.shape == (6, 5)
    assert result.affine.a.shape == (5, 5)
    assert result.affine.diagonal is True  # B = 6 < 2k = 10, Sec. 3.1

    big = model.adapt(features[:64])
    assert big.k == 11
    assert big.affine.a.shape == (11, 11)
    assert big.affine.diagonal is False


def test_lda_directions_solve_the_generalized_eigenproblem():
    """S_b u = lambda S_w u for the returned directions, to 1e-8 relative.

    The residual is checked against the unregularized scatters, so the
    within-class ridge shows up in it. The directions are also S_w-orthonormal
    by construction, which is what makes the projected Sigma_s close to I.
    """
    key = jax.random.PRNGKey(2)
    num_classes, dim = 8, 16
    params = class_parameters(key, num_classes, dim, separation=3.0)
    features, labels = sample_classes(jax.random.PRNGKey(3), params, 500)
    s_b, s_w, _, _, _ = scatter_matrices(features, labels, num_classes)
    directions = lda_directions(s_b, s_w, k0=7, within_ridge=1e-10)

    gram = directions.T @ s_w @ directions
    assert np.max(np.abs(np.asarray(gram - jnp.eye(7)))) < 1e-8

    for j in range(7):
        u = directions[:, j]
        lhs, rhs = s_b @ u, s_w @ u
        eigenvalue = float((u @ lhs) / (u @ rhs))
        residual = np.linalg.norm(np.asarray(lhs - eigenvalue * rhs))
        assert residual < 1e-8 * float(np.linalg.norm(np.asarray(lhs)))


def test_lda_projection_is_invariant_to_invertible_feature_rescaling():
    """Rescaling the raw features leaves z = U_k^T h alone.

    Under h -> L h the scatters transform as S -> L S L^T, so the generalized
    eigenvectors transform as u -> L^-T u and the projection u^T h does not
    move, up to the sign of each direction. This is the "invariant to feature
    conditioning" property Sec. 3.2 leans on. It is exact only for zero
    within-class ridge, so the test uses a tiny one and a moderately
    conditioned L.
    """
    key = jax.random.PRNGKey(4)
    num_classes, dim = 8, 12
    params = class_parameters(key, num_classes, dim, separation=3.0)
    features, labels = sample_classes(jax.random.PRNGKey(5), params, 400)

    scales = jnp.exp(jax.random.uniform(jax.random.PRNGKey(6), (dim,), dtype=jnp.float64) * 2 - 1)
    rotation, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(7), (dim, dim), dtype=jnp.float64))
    lmap = rotation * scales[None, :]

    config = CasterConfig(within_ridge=1e-12)
    plain = Caster.fit(features, labels, num_classes, config)
    rescaled = Caster.fit(features @ lmap.T, labels, num_classes, config)

    z_plain = np.asarray(plain.project(features))
    z_rescaled = np.asarray(rescaled.project(features @ lmap.T))
    # The sign of each direction is a gauge: flipping column j flips coordinate
    # j of z and of every stored moment, so nothing downstream sees it. The
    # deterministic sign rule in lda_directions picks its pivot in the original
    # coordinates, so L can flip it. Align, then compare.
    signs = np.sign(np.sum(z_plain * z_rescaled, axis=0))
    scale = np.max(np.abs(z_plain))
    assert np.max(np.abs(z_plain - z_rescaled * signs[None, :])) < 1e-7 * scale
