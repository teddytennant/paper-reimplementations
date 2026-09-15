import numpy as np

from flr_noisy.basis import (
    discrete_inner_product,
    fourier_basis_matrix,
    lemma_a1_inner_product,
    theoretical_recon_dim,
)


def test_discrete_orthonormality_when_dim_lt_p():
    p = 32
    n_basis = 11
    phi = np.asarray(fourier_basis_matrix(p, n_basis))
    gram = (phi @ phi.T) / p
    np.testing.assert_allclose(gram, np.eye(n_basis), atol=1e-12)


def test_lemma_a1_matches_grid_inner_product():
    p = 16
    n_basis = 15
    phi = np.asarray(fourier_basis_matrix(p, n_basis))
    for l in range(1, n_basis + 1):
        for j in range(1, n_basis + 1):
            grid = float(discrete_inner_product(phi[l - 1], phi[j - 1]))
            closed = lemma_a1_inner_product(l, j, p)
            assert np.isclose(grid, closed, atol=1e-12)
            if l < p and j < p:
                assert np.isclose(closed, float(l == j), atol=1e-12)


def test_lemma_a1_aliasing_example():
    p = 8
    # l = j = 1
    assert lemma_a1_inner_product(1, 1, p) == 1.0
    # both even, no aliasing for l, j < p
    assert lemma_a1_inner_product(2, 2, p) == 1.0
    assert lemma_a1_inner_product(2, 4, p) == 0.0


def test_theoretical_recon_dim_odd_and_below_cap():
    d = theoretical_recon_dim(80, 64)
    assert d % 2 == 1
    assert d < min(80 / (np.log(80) ** 2), 64)
    assert d >= 1
