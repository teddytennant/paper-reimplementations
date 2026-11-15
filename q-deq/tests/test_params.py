"""Shared DEQ layer has fewer params than an explicit stack of the same layer."""

from q_deq.deq import count_params, init_shared_params, init_stacked_params
import jax


def test_shared_layer_param_count_vs_explicit_stack():
    key = jax.random.PRNGKey(14)
    hidden, input_dim, mlp_hidden = 16, 24, 16
    n_layers = 3
    shared = init_shared_params(
        key, hidden=hidden, input_dim=input_dim, mlp_hidden=mlp_hidden
    )
    stacked = init_stacked_params(
        key, n_layers, hidden=hidden, input_dim=input_dim, mlp_hidden=mlp_hidden
    )
    n_shared = count_params(shared)
    n_stack = count_params(stacked)
    assert n_shared > 0
    assert n_stack == n_layers * n_shared
    assert n_stack / n_shared == n_layers
