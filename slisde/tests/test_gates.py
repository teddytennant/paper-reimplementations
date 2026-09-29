"""Identity / zero gates reduce to independent stacked linear layers (Eqs. 7, 81, 82)."""

import jax
import jax.numpy as jnp

from slisde.gates import apply_inflow, gated_pairs, scale_and_offset, scale_diag, zero_gates
from slisde.model import init_model, sample_z0, simulate
from slisde.sde import euler_maruyama_pairs, init_layer, sample_increments
from slisde.scan import sequential_simulate


def test_identity_modulation_leaves_pairs_unchanged():
    F = jnp.array([[[[1.2, 0.3], [-0.1, 0.8]]]])
    g = jnp.array([[[0.4, -0.2]]])
    alpha = jnp.ones((1, 1, 2))
    offset = jnp.zeros((1, 1, 2))
    Fb, gb = apply_inflow(F, g, alpha, offset)
    assert jnp.allclose(Fb, F)
    assert jnp.allclose(gb, g)


def test_zero_glu_is_identity_inflow():
    d, x_dim = 3, 7
    gates = zero_gates(d, x_dim)
    x = jnp.ones((2, 5, x_dim))
    alpha, offset = scale_and_offset(gates, x)
    assert jnp.allclose(alpha, 1.0)
    assert jnp.allclose(offset, 0.0)


def test_scale_diag_only_touches_diagonal():
    F = jnp.arange(9.0).reshape(3, 3)
    alpha = jnp.array([2.0, 0.5, 1.5])
    Fb = scale_diag(F, alpha)
    assert jnp.allclose(jnp.diagonal(Fb), jnp.diagonal(F) * alpha)
    off = ~jnp.eye(3, dtype=bool)
    assert jnp.allclose(Fb[off], F[off])


def test_zero_gates_stack_matches_independent_linear_layers():
    key = jax.random.PRNGKey(2)
    B, N, d, m = 6, 8, 3, 1
    dt = 0.1
    k1, k2, kz, kW, kWp = jax.random.split(key, 5)
    layer0 = init_layer(k1, d, m, noise="additive")
    layer1 = init_layer(k2, d, m, noise="additive")
    z0 = sample_z0(kz, B, d)
    dW0 = sample_increments(kW, B, N, m, dt)
    dW1 = sample_increments(kWp, B, N, m, dt)
    times = jnp.linspace(0.0, dt * N, N + 1)

    F0, g0 = euler_maruyama_pairs(layer0["A"], layer0["b"], layer0["C"], layer0["D"], dW0, dt)
    path0 = sequential_simulate(F0, g0, z0)
    F1, g1 = euler_maruyama_pairs(layer1["A"], layer1["b"], layer1["C"], layer1["D"], dW1, dt)
    gates = zero_gates(d, d + 4)
    Fg, gg = gated_pairs(gates, F1, g1, path0[:, :-1, :], times[:-1])
    assert jnp.allclose(Fg, F1, atol=1e-6)
    assert jnp.allclose(gg, g1, atol=1e-6)
    path1_gated = sequential_simulate(Fg, gg, z0)
    path1_indep = sequential_simulate(F1, g1, z0)
    assert jnp.allclose(path1_gated, path1_indep, atol=1e-6)


def test_zero_gates_ignore_previous_path():
    key = jax.random.PRNGKey(3)
    B, N, d = 4, 6, 2
    dt = 0.1
    kL, kz, kW = jax.random.split(key, 3)
    params = init_model(kL, n_layers=2, d=d, m=1, p=1, gate_scale=0.0)
    params["gates"] = [zero_gates(d, d + 4)]
    z0 = sample_z0(kz, B, d)
    dW = sample_increments(kW, B, N, 1, dt)
    Y_a, _, _ = simulate(params, z0, dW, dt)
    params_b = {
        **params,
        "layers": [
            {**params["layers"][0], "b": params["layers"][0]["b"] + 1.5},
            params["layers"][1],
        ],
    }
    Y_b, _, _ = simulate(params_b, z0, dW, dt)
    # last layer uses the same z0 and same dW; identity gates ignore the previous path
    assert jnp.allclose(Y_a, Y_b, atol=1e-5)

    params_c = init_model(kL, n_layers=2, d=d, m=1, p=1, gate_scale=1.0)
    Y_c, _, _ = simulate(params_c, z0, dW, dt)
    params_d = {
        **params_c,
        "layers": [
            {**params_c["layers"][0], "b": params_c["layers"][0]["b"] + 1.5},
            params_c["layers"][1],
        ],
    }
    Y_d, _, _ = simulate(params_d, z0, dW, dt)
    assert not jnp.allclose(Y_c, Y_d, atol=1e-4)
