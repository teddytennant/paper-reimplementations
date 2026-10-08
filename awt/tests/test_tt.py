"""TT-SVD reconstruction, rank shapes, and matvec against the dense oracle."""

import inspect

import jax.numpy as jnp

from awt.tt import densify, tt_matvec, tt_svd


def _interleave_k2(matrix):
    """Spec mode order for factors (2, 2) and (2, 2): C reshape, then (0, 2, 1, 3)."""
    reshaped = jnp.reshape(matrix, (2, 2, 2, 2))
    return jnp.transpose(reshaped, (0, 2, 1, 3))


def test_full_rank_reconstructs_known_matrix_and_rank1_shapes():
    matrix = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    interleaved = _interleave_k2(matrix)
    # One entry of the interleaved tensor, to lock C-order plus the transpose.
    # row = i1*2+i2, col = j1*2+j2, so (i1, j1, i2, j2) = (0, 1, 0, 1) is W[0, 3].
    assert jnp.allclose(interleaved[0, 1, 0, 1], matrix[0, 3])

    cores = tt_svd(interleaved, ranks=None)
    assert cores[0].shape == (1, 2, 2, 4)
    assert cores[1].shape == (4, 2, 2, 1)
    recon = densify(cores)
    assert recon.shape == (4, 4)
    assert jnp.allclose(recon, matrix, atol=1e-4, rtol=1e-4)

    rank1 = tt_svd(interleaved, ranks=[1])
    assert rank1[0].shape == (1, 2, 2, 1)
    assert rank1[1].shape == (1, 2, 2, 1)
    assert densify(rank1).shape == (4, 4)

    clamped = tt_svd(interleaved, ranks=[100])
    assert clamped[0].shape == (1, 2, 2, 4)
    assert clamped[1].shape == (4, 2, 2, 1)


def test_matvec_matches_dense_oracle_and_does_not_call_densify():
    matrix = jnp.arange(16.0).reshape(4, 4)
    cores = tt_svd(_interleave_k2(matrix), ranks=None)
    x = jnp.array([0.5, -1.0, 2.0, 0.25])
    y = tt_matvec(cores, x)
    y_ref = densify(cores) @ x
    assert y.shape == (4,)
    assert jnp.allclose(y, y_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(y, matrix @ x, atol=1e-4, rtol=1e-4)
    assert "densify" not in inspect.getsource(tt_matvec)


def test_k3_full_rank_and_matvec():
    matrix = jnp.arange(64.0).reshape(8, 8) + jnp.eye(8)
    reshaped = jnp.reshape(matrix, (2, 2, 2, 2, 2, 2))
    # (m1, m2, m3, n1, n2, n3) -> (m1, n1, m2, n2, m3, n3)
    interleaved = jnp.transpose(reshaped, (0, 3, 1, 4, 2, 5))
    cores = tt_svd(interleaved, ranks=None)
    assert [tuple(c.shape) for c in cores] == [
        (1, 2, 2, 4),
        (4, 2, 2, 4),
        (4, 2, 2, 1),
    ]
    assert jnp.allclose(densify(cores), matrix, atol=1e-4, rtol=1e-4)
    x = jnp.arange(8.0)
    assert jnp.allclose(tt_matvec(cores, x), matrix @ x, atol=1e-4, rtol=1e-4)


def test_nonsquare_full_rank():
    matrix = jnp.arange(24.0).reshape(6, 4)
    reshaped = jnp.reshape(matrix, (2, 3, 2, 2))
    interleaved = jnp.transpose(reshaped, (0, 2, 1, 3))
    cores = tt_svd(interleaved, ranks=None)
    assert cores[0].shape[1:3] == (2, 2)
    assert cores[1].shape[1:3] == (3, 2)
    assert jnp.allclose(densify(cores), matrix, atol=1e-4, rtol=1e-4)
    x = jnp.array([1.0, -1.0, 0.5, 0.25])
    assert jnp.allclose(tt_matvec(cores, x), matrix @ x, atol=1e-4, rtol=1e-4)
