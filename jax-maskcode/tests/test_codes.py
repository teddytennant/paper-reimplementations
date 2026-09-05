"""Generator, parity-check, Tanner mask, and soft syndrome identities."""

import jax.numpy as jnp
import numpy as np
import pytest

from maskcode import codes, gf2, ops


ALL_CODES = [
    codes.hamming74(),
    codes.single_parity_check(4),
    codes.bch(4, 2),
    codes.array_ldpc(5, 2, 4),
    codes.tree_code(),
]


@pytest.mark.parametrize("code", ALL_CODES, ids=lambda c: c.name)
def test_generator_is_orthogonal_to_parity_checks(code):
    assert not gf2.matmul(code.g, code.h.T).any()


@pytest.mark.parametrize("code", ALL_CODES, ids=lambda c: c.name)
def test_every_codeword_has_zero_hard_syndrome(code):
    if code.k > 10:
        rng = np.random.default_rng(0)
        m = rng.integers(0, 2, size=(64, code.k), dtype=np.uint8)
        cw = code.encode(m)
    else:
        cw = code.codewords()
    assert not code.hard_syndrome(cw).any()


def test_flipping_a_bit_fails_every_check_that_touches_it():
    code = codes.hamming74()
    c = code.encode(np.array([[1, 0, 1, 1]], dtype=np.uint8))[0]
    assert not code.hard_syndrome(c).any()
    for v in range(code.n):
        flipped = c.copy()
        flipped[v] ^= 1
        syn = code.hard_syndrome(flipped)
        for j in range(code.n_checks):
            if code.h[j, v]:
                assert syn[j] == 1
            else:
                assert syn[j] == 0


def test_attention_mask_follows_equation_23_and_24():
    code = codes.hamming74()
    a = code.tanner_adjacency()
    n, m = code.n, code.n_checks
    gram = (code.h.T @ code.h > 0).astype(np.uint8)
    assert np.array_equal(a[:n, :n], gram)
    assert np.array_equal(a[:n, n:], code.h.T)
    assert np.array_equal(a[n:, :n], code.h)
    assert np.array_equal(a[n:, n:], np.eye(m, dtype=np.uint8))
    mask = np.asarray(code.attention_mask())
    assert np.all(mask[a == 1] == 0.0)
    assert np.all(np.isneginf(mask[a == 0]))


def test_soft_syndrome_of_a_noiseless_even_degree_codeword_is_positive():
    """Hamming checks have even weight, so a valid codeword saturates s_i > 0."""
    code = codes.hamming74()
    c = code.encode(np.array([[1, 1, 0, 1]], dtype=np.uint8))[0]
    bipolar = 2.0 * c.astype(np.float64) - 1.0
    # Large |r| in the paper's log(p1/p0) convention: r = k * (2c-1)
    r = 20.0 * jnp.asarray(bipolar)
    s = np.asarray(code.soft_syndrome(r))
    assert np.all(s > 8.0)


def test_soft_syndrome_sign_on_an_odd_degree_check():
    """tree_code rows have weight 3: sign(s) -> (-1)^(d + syn) = -(-1)^syn."""
    code = codes.tree_code()
    c = code.encode(np.zeros((1, code.k), dtype=np.uint8))[0]
    r = 20.0 * (2.0 * jnp.asarray(c, dtype=jnp.float64) - 1.0)
    s = np.asarray(code.soft_syndrome(r))
    # syn = 0, d odd, so sign is -1
    assert np.all(s < -8.0)
