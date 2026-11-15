import jax
import jax.numpy as jnp

from learnable_subspace import (
    dense_projector,
    fit_mlp,
    init_mlp,
    merge_input,
    orthogonal_factor,
)


def test_frozen_mlp_loss_decreases_and_merge_reconstructs():
    model, x = init_mlp(jax.random.PRNGKey(0), width=16, k=4, batch=32)
    W1, b1, W2, b2 = model.W1, model.b1, model.W2, model.b2
    V1_0, V2_0 = model.V1, model.V2

    model, losses = fit_mlp(model, x, steps=40, lr=1e-2, lambda_ort=1e-3)

    assert losses.shape == (41,)
    assert bool(jnp.all(jnp.isfinite(losses)))
    assert float(losses[-1]) < float(losses[0])

    assert jnp.array_equal(model.W1, W1)
    assert jnp.array_equal(model.W2, W2)
    assert jnp.array_equal(model.b1, b1)
    assert jnp.array_equal(model.b2, b2)
    assert not jnp.allclose(model.V1, V1_0)
    assert not jnp.allclose(model.V2, V2_0)

    for W, V in ((model.W1, model.V1), (model.W2, model.V2)):
        U = orthogonal_factor(V)
        A, B = merge_input(W, U)
        P = dense_projector(U)
        err = jnp.max(jnp.abs(W @ P - B @ A))
        assert float(err) < 1e-5
        assert A.shape[0] == W.shape[1] - V.shape[1]
