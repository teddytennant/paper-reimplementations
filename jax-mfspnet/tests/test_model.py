"""Dense blocks (Eq. 1) and stacked aggregation (Eqs. 9-11), including the
gradient behaviour the stacking design is justified by (Sec. 4.6, Sec. 6.8).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfspnet.model import (
    PSI,
    T_MAX,
    StackConfig,
    aggregation_weights,
    apply_dense_block,
    apply_network,
    block_output_channels,
    count_parameters,
    init_dense_block,
    init_network,
)


def test_table_2_stacking_constants():
    assert PSI == 0.88
    assert T_MAX == 8


def test_aggregation_weights_are_equation_10():
    """alpha_i = exp(-psi (l-i)) / sum_k exp(-psi (l-k)), normalized."""
    l, psi = 5, PSI
    a = aggregation_weights(l, psi)
    raw = np.array([np.exp(-psi * (l - i)) for i in range(1, l)])
    assert np.allclose(a, raw / raw.sum())
    assert a.sum() == pytest.approx(1.0)


def test_aggregation_weights_favour_the_nearest_block():
    a = aggregation_weights(6, PSI)
    assert np.all(np.diff(a) > 0)  # ascending in i, so nearest is heaviest
    assert a[-1] / a[0] == pytest.approx(np.exp(PSI * 4))
    assert aggregation_weights(1, PSI).size == 0


def test_a_larger_psi_concentrates_weight_on_the_nearest_block():
    assert aggregation_weights(6, 2.0)[-1] > aggregation_weights(6, 0.2)[-1]
    assert np.allclose(aggregation_weights(6, 1e-9), np.full(5, 0.2))


def test_dense_block_widths_follow_equation_1():
    """N_{l+1} = sum_{j=0..l} k_j, block input included."""
    growth = [3, 5, 2]
    params = init_dense_block(jax.random.key(0), 4, growth, None)
    widths = [layer["conv"].shape[2] for layer in params["layers"]]
    assert widths == [4, 4 + 3, 4 + 3 + 5]
    out = apply_dense_block(params, jnp.zeros((2, 8, 8, 4)))
    assert out.shape[-1] == 4 + 3 + 5 + 2
    assert block_output_channels(4, growth, None) == 14
    assert block_output_channels(4, growth, 6) == 6


def test_block_output_carries_the_block_input_through():
    """Sec. 2.1: the block output concatenates every layer output with the
    original input feature maps, so the input is still there at the end."""
    params = init_dense_block(jax.random.key(0), 3, [2, 2], None)
    x = jax.random.normal(jax.random.key(1), (2, 6, 6, 3))
    out = apply_dense_block(params, x)
    assert jnp.allclose(out[..., :3], x)


def test_empty_block_is_the_identity():
    params = init_dense_block(jax.random.key(0), 5, [], None)
    x = jax.random.normal(jax.random.key(1), (2, 6, 6, 5))
    assert jnp.allclose(apply_dense_block(params, x), x)


def test_logits_shape_and_parameter_growth_with_stack_depth():
    cfg1 = StackConfig(n_classes=7, stem_channels=8, n_blocks=1)
    cfg3 = StackConfig(n_classes=7, stem_channels=8, n_blocks=3)
    p1 = init_network(jax.random.key(0), [4, 4], 3, cfg1)
    p3 = init_network(jax.random.key(0), [4, 4], 3, cfg3)
    x = jnp.zeros((2, 8, 8, 3))
    assert apply_network(p1, x, cfg1).shape == (2, 7)
    assert apply_network(p3, x, cfg3).shape == (2, 7)
    assert count_parameters(p3) > count_parameters(p1)


def test_projection_cost_is_the_only_extra_over_plain_stacking():
    """Sec. 6.8: the 1x1 projections keep the stack near constant width, so the
    aggregation adds a small fraction of the parameters, not a multiple."""
    dense = StackConfig(n_classes=4, stem_channels=12, n_blocks=6)
    plain = StackConfig(n_classes=4, stem_channels=12, n_blocks=6, dense_stacking=False)
    pd = count_parameters(init_network(jax.random.key(0), [6, 6, 6], 3, dense))
    pp = count_parameters(init_network(jax.random.key(0), [6, 6, 6], 3, plain))
    assert pp < pd < 1.35 * pp


@functools.lru_cache(maxsize=None)
def _stack_gradients(dense, n_blocks, seed=0, width=12, growth=(4, 4, 4)):
    """Relative gradient norm ||dL/dW|| / ||W|| at the first conv of every block."""
    cfg = StackConfig(
        n_classes=4, stem_channels=width, n_blocks=n_blocks, dense_stacking=dense
    )
    params = init_network(jax.random.key(seed), list(growth), 3, cfg)
    x = jax.random.normal(jax.random.key(99), (8, 8, 8, 3))
    y = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])

    def loss(p):
        logits = apply_network(p, x, cfg)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(8), y])

    grads = jax.grad(loss)(params)
    return np.array(
        [
            float(
                jnp.linalg.norm(g["layers"][0]["conv"])
                / jnp.linalg.norm(b["layers"][0]["conv"])
            )
            for b, g in zip(params["blocks"], grads["blocks"])
        ]
    )


def test_dense_stacking_keeps_the_gradient_flat_across_a_deep_stack():
    """The reason Eq. 11 exists (Sec. 4.6, Sec. 6.8: "stabilizing gradient
    propagation"). Measured at initialization on a 12-block stack.

    Direction is worth stating plainly, because it is not the textbook one.
    Every composite layer here is BN-ReLU-Conv, so the plain stack does not
    lose its first-layer gradient; it blows it up. At 12 blocks the plain
    stack's front-block gradient runs about 40 times its last block's, and the
    aggregation pulls that spread down several fold while leaving the first
    layer a usable gradient. Either way the failure is that the front and the
    back of the network end up on wildly different scales, and the aggregation
    is what fixes it.
    """
    shallow_dense = _stack_gradients(True, 2)
    shallow_plain = _stack_gradients(False, 2)
    deep_dense = _stack_gradients(True, 12)
    deep_plain = _stack_gradients(False, 12)

    spread_dense = deep_dense[0] / deep_dense[-1]
    spread_plain = deep_plain[0] / deep_plain[-1]

    # both are fine when the stack is shallow
    assert shallow_dense[0] / shallow_dense[-1] < 4.0
    assert shallow_plain[0] / shallow_plain[-1] < 4.0

    # at depth the plain stack comes apart and the aggregated one does not
    assert spread_plain > 25.0
    assert spread_dense < 15.0
    assert spread_plain > 3.0 * spread_dense

    # the first layer of the aggregated stack is neither vanished nor blown up
    assert 0.05 < deep_dense[0] < 3.0
    assert deep_plain[0] > 2.5 * deep_dense[0]


def test_the_gradient_spread_of_a_plain_stack_grows_with_depth():
    """Same measurement as a function of depth: the plain stack's spread runs
    away, the aggregated one levels off."""
    depths = (2, 8, 12)
    plain = [_stack_gradients(False, d) for d in depths]
    dense = [_stack_gradients(True, d) for d in depths]
    spread_plain = [g[0] / g[-1] for g in plain]
    spread_dense = [g[0] / g[-1] for g in dense]
    assert spread_plain[0] < spread_plain[1] < spread_plain[2]
    assert spread_plain[2] / spread_plain[0] > 5.0 * (spread_dense[2] / spread_dense[0])


def test_aggregation_actually_changes_the_forward_pass():
    """Guard against the aggregation branch being silently dead."""
    dense = StackConfig(n_classes=4, stem_channels=8, n_blocks=3)
    plain = StackConfig(n_classes=4, stem_channels=8, n_blocks=3, dense_stacking=False)
    params = init_network(jax.random.key(0), [4, 4], 3, dense)
    x = jax.random.normal(jax.random.key(1), (2, 8, 8, 3))
    assert not jnp.allclose(
        apply_network(params, x, dense), apply_network(params, x, plain)
    )


def test_forward_pass_is_finite_at_the_paper_stack_depth():
    """T_max = 8 (Table 2)."""
    cfg = StackConfig(n_classes=10, stem_channels=12, n_blocks=T_MAX)
    params = init_network(jax.random.key(0), [12, 16, 12], 3, cfg)
    out = apply_network(params, jax.random.normal(jax.random.key(1), (4, 8, 8, 3)), cfg)
    assert out.shape == (4, 10)
    assert bool(jnp.isfinite(out).all())
