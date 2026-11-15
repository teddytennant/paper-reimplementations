"""Section 4.1: the attention map as alpha * v_ij I_b + (1 - alpha) * c_ij J_b,
and the b strided heads plus one downsampled head that compute it."""

import jax
import jax.numpy as jnp
import pytest

from gale import ALPHA_DEFAULT, attention, attention_map, effective_attention_matrix, gale_attention
from gale.attention import block_mean, strided

F64 = jnp.float64


def qkv(seed, n=12, d=4):
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    return tuple(jax.random.normal(k, (n, d), dtype=F64) for k in ks)


@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
def test_b_one_is_exact_attention(alpha):
    """At b = 1 there is one strided head covering every token and the downsample
    is the identity, so Equation 4 collapses to Equation 2 for any alpha."""
    q, k, v = qkv(0)
    out = gale_attention(q, k, v, b=1, alpha=alpha)
    assert float(jnp.max(jnp.abs(out - attention(q, k, v)))) < 1e-14


@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("alpha", [0.6, ALPHA_DEFAULT, 1.0])
def test_output_equals_its_own_block_structured_map_applied_to_v(b, alpha):
    """Equation 4 computed head by head is the same thing as Equation 3's block
    structured map multiplied into V. Nothing else is going on."""
    q, k, v = qkv(1)
    out = gale_attention(q, k, v, b=b, alpha=alpha)
    a = effective_attention_matrix(q, k, b=b, alpha=alpha)
    assert float(jnp.max(jnp.abs(out - a @ v))) < 1e-13


@pytest.mark.parametrize("b", [2, 3])
def test_blocks_are_the_paper_s_identity_plus_all_ones(b):
    """Constructed case where Equation 3 holds exactly.

    Make Q and K constant inside every block of b consecutive tokens. Then all b
    strided heads see the same scores, so the local part of block (i, j) really
    is v_ij I_b rather than a diagonal with a value per phase, and the
    downsampled head reproduces c_ij J_b. Assembling the blocks by hand from
    Equation 3 has to match what the implementation computes.
    """
    n, d = 12, 4
    nb = n // b
    ks = jax.random.split(jax.random.PRNGKey(2), 3)
    q = jnp.repeat(jax.random.normal(ks[0], (nb, d), dtype=F64), b, axis=0)
    k = jnp.repeat(jax.random.normal(ks[1], (nb, d), dtype=F64), b, axis=0)
    v = jax.random.normal(ks[2], (n, d), dtype=F64)
    alpha = ALPHA_DEFAULT

    coarse = attention_map(block_mean(q, b), block_mean(k, b))
    eye, ones = jnp.eye(b, dtype=F64), jnp.ones((b, b), dtype=F64)
    by_hand = jnp.block(
        [
            [
                alpha * coarse[i, j] * eye + (1 - alpha) * (coarse[i, j] / b) * ones
                for j in range(nb)
            ]
            for i in range(nb)
        ]
    )
    built = effective_attention_matrix(q, k, b=b, alpha=alpha)
    assert float(jnp.max(jnp.abs(built - by_hand))) < 1e-14
    assert float(jnp.max(jnp.abs(gale_attention(q, k, v, b=b, alpha=alpha) - by_hand @ v))) < 1e-13
    # v_ij here is the softmax weight of the coarse map and c_ij is that weight
    # spread over b columns, so the two components carry the same total mass.
    assert float(jnp.sum(by_hand)) == pytest.approx(n, abs=1e-12)


@pytest.mark.parametrize("b", [2, 3, 4])
def test_local_part_only_connects_tokens_of_the_same_phase(b):
    """L_ij = v_ij I_b keeps the diagonal of each block and nothing else, which
    is the same statement as: token t only sees tokens congruent to t mod b."""
    q, k, _ = qkv(3)
    a = effective_attention_matrix(q, k, b=b, alpha=1.0)
    n = q.shape[0]
    idx = jnp.arange(n)
    same_phase = (idx[:, None] - idx[None, :]) % b == 0
    assert float(jnp.max(jnp.abs(jnp.where(same_phase, 0.0, a)))) == 0.0
    assert float(jnp.min(jnp.abs(jnp.where(same_phase, a, 1.0)))) > 0.0
    assert jnp.allclose(a.sum(axis=1), 1.0, atol=1e-13)


@pytest.mark.parametrize("b", [2, 3])
def test_strided_heads_are_the_phases_and_the_downsample_is_a_block_mean(b):
    q, _, _ = qkv(4)
    n = q.shape[0]
    for phase in range(b):
        assert jnp.array_equal(strided(q, b, phase), q[phase::b])
    assert block_mean(q, b).shape == (n // b, q.shape[1])
    assert jnp.allclose(block_mean(q, b)[0], q[:b].mean(axis=0))


def test_global_part_is_constant_inside_every_block():
    """G_ij = c_ij J_b: with alpha = 0 the map depends only on the block indices."""
    b, alpha = 3, 0.0
    q, k, _ = qkv(5)
    a = effective_attention_matrix(q, k, b=b, alpha=alpha)
    n = q.shape[0]
    blocks = a.reshape(n // b, b, n // b, b)
    spread = jnp.max(blocks, axis=(1, 3)) - jnp.min(blocks, axis=(1, 3))
    assert float(jnp.max(spread)) < 1e-15


def test_approximation_error_grows_with_b():
    """b trades memory for fidelity: the map shrinks as (N/b)^2 and the error
    goes the other way. Averaged over seeds, since which of two nearby b values
    wins on one random Q, K is a coin flip.
    """
    bs = (1, 2, 3, 4, 6)
    curves = []
    for seed in range(8):
        q, k, v = qkv(100 + seed, n=24, d=4)
        ref = attention(q, k, v)
        curves.append(
            [
                float(jnp.mean((gale_attention(q, k, v, b=b, alpha=ALPHA_DEFAULT) - ref) ** 2))
                for b in bs
            ]
        )
    mean = [sum(c[i] for c in curves) / len(curves) for i in range(len(bs))]
    assert mean[0] < 1e-28
    assert all(a < b for a, b in zip(mean[:-1], mean[1:])), mean
