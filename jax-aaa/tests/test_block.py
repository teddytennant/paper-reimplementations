"""AAA block residual shape and two-layer grid connectivity."""

import jax
import jax.numpy as jnp

from aaa.block import aaa_block, init_block_params
from aaa.model import axon_encode, init_axon_params


def test_block_output_shape():
    key = jax.random.PRNGKey(0)
    b, c, t, d, h = 2, 3, 4, 8, 2
    x = jax.random.normal(key, (b, c, t, d))
    params = init_block_params(jax.random.PRNGKey(1), d, gate="gated")
    out = aaa_block(params, x, num_heads=h)
    assert out.shape == (b, c, t, d)


def test_fixed_block_output_shape():
    key = jax.random.PRNGKey(2)
    b, c, t, d = 1, 5, 6, 16
    x = jax.random.normal(key, (b, c, t, d))
    params = init_block_params(jax.random.PRNGKey(3), d, gate="fixed")
    out = aaa_block(params, x, num_heads=4)
    assert out.shape == (b, c, t, d)


def _jac_token_pair(apply_fn, x, c_out, t_out, c_in, t_in):
    """Frobenius norm of d output[0,c_out,t_out] / d input[0,c_in,t_in]."""

    def scalar_map(src):
        xx = x.at[0, c_in, t_in].set(src)
        y = apply_fn(xx)
        return y[0, c_out, t_out]

    jac = jax.jacrev(scalar_map)(x[0, c_in, t_in])
    return float(jnp.linalg.norm(jac))


def test_two_blocks_connect_off_axis_tokens():
    """Appendix E.1: two layers connect any (c,t) to any (c',t')."""
    key = jax.random.PRNGKey(10)
    b, c, t, d, heads = 1, 3, 4, 8, 2
    x = jax.random.normal(key, (b, c, t, d))
    params = init_axon_params(
        jax.random.PRNGKey(11),
        c=c,
        t=t,
        d=d,
        num_blocks=2,
        gate="fixed",
    )

    def encode(xx):
        return axon_encode(params, xx, num_heads=heads)

    # off-axis pair: different channel and different time
    nrm = _jac_token_pair(encode, x, c_out=2, t_out=3, c_in=0, t_in=0)
    assert nrm > 1e-6


def test_single_path_block_not_fully_connected():
    """One temporal-only (or spatial-only) block cannot mix off-axis tokens."""
    key = jax.random.PRNGKey(12)
    b, c, t, d, heads = 1, 3, 4, 8, 2
    x = jax.random.normal(key, (b, c, t, d))
    p_t = init_block_params(jax.random.PRNGKey(13), d, gate="temporal")
    p_s = init_block_params(jax.random.PRNGKey(14), d, gate="spatial")

    def temporal_only(xx):
        return aaa_block(p_t, xx, num_heads=heads, path="temporal")

    def spatial_only(xx):
        return aaa_block(p_s, xx, num_heads=heads, path="spatial")

    n_t = _jac_token_pair(temporal_only, x, c_out=2, t_out=3, c_in=0, t_in=0)
    n_s = _jac_token_pair(spatial_only, x, c_out=2, t_out=3, c_in=0, t_in=0)
    assert n_t < 1e-6
    assert n_s < 1e-6
    # same-channel different-time is reachable in one temporal step
    n_t_same_c = _jac_token_pair(
        temporal_only, x, c_out=0, t_out=3, c_in=0, t_in=0
    )
    assert n_t_same_c > 1e-6
    # same-time different-channel is reachable in one spatial step
    n_s_same_t = _jac_token_pair(
        spatial_only, x, c_out=2, t_out=0, c_in=0, t_in=0
    )
    assert n_s_same_t > 1e-6


def test_one_mixed_block_also_not_off_axis():
    """A single mixed block still cannot jump off-axis in one step."""
    key = jax.random.PRNGKey(15)
    b, c, t, d, heads = 1, 3, 4, 8, 2
    x = jax.random.normal(key, (b, c, t, d))
    params = init_block_params(jax.random.PRNGKey(16), d, gate="gated")

    def one_block(xx):
        return aaa_block(params, xx, num_heads=heads)

    n_off = _jac_token_pair(one_block, x, c_out=2, t_out=3, c_in=0, t_in=0)
    assert n_off < 1e-6
