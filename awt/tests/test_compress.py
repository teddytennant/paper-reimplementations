"""Column preconditioning identity and inverse-scale deployment."""

import jax.numpy as jnp

from awt.compress import compress, deploy
from awt.tt import densify, tt_matvec


def test_full_rank_column_scale_identity():
    W = jnp.array(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.5, 1.5, 0.0, 1.0],
            [-1.0, 2.0, 0.5, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    C = jnp.array(
        [
            [1.0, 0.1, 2.0, 0.05],
            [1.2, 0.2, 1.5, 0.02],
            [0.8, 0.05, 2.5, 0.01],
        ]
    )
    scales, cores = compress(
        W,
        C,
        "std",
        1.0,
        1e-6,
        1e-3,
        (2, 2),
        (2, 2),
        None,
    )
    assert scales.shape == (4,)
    # Higher-variance channels are not required to beat SVD; only the identity.
    w_hat_prime = densify(cores)
    w_prime = W * scales[None, :]
    assert jnp.allclose(w_hat_prime, w_prime, atol=1e-4, rtol=1e-4)

    # ||(W - W_hat' diag(1/s)) diag(s) - (W' - W_hat')|| ~ 0
    inv_s = 1.0 / scales
    left = (W - w_hat_prime * inv_s[None, :]) * scales[None, :]
    right = w_prime - w_hat_prime
    err = jnp.linalg.norm(left - right)
    assert float(err) < 1e-4

    # Row scaling would not match column-scaled W'.
    row_scaled = scales[:, None] * W
    assert not jnp.allclose(w_hat_prime, row_scaled, atol=1e-3, rtol=1e-3)


def test_deploy_applies_inverse_scales():
    W = jnp.arange(16.0).reshape(4, 4) + 0.5
    C = jnp.array(
        [
            [2.0, 0.2, 1.0, 0.05],
            [1.5, 0.1, 1.2, 0.02],
            [2.5, 0.3, 0.8, 0.01],
        ]
    )
    scales, cores = compress(
        W,
        C,
        statistic="mean_abs",
        alpha=1.5,
        eps=1e-6,
        min_scale=1e-3,
        m_factors=(2, 2),
        n_factors=(2, 2),
        ranks=None,
    )
    assert not jnp.allclose(scales, jnp.ones_like(scales), atol=1e-3)

    x = jnp.array([1.0, -2.0, 0.5, 3.0])
    y = deploy(cores, scales, x)
    dense = densify(cores)
    y_ref = dense @ (x / scales)
    assert jnp.allclose(y, y_ref, atol=1e-4, rtol=1e-4)

    # Inverse scale is actually applied: the deployed product is not W_hat' @ x.
    y_unscaled = dense @ x
    assert not jnp.allclose(y, y_unscaled, atol=1e-4, rtol=1e-4)
    # The bare matvec is the unscaled product. deploy is what divides by s.
    assert jnp.allclose(tt_matvec(cores, x), y_unscaled, atol=1e-4, rtol=1e-4)
    assert not jnp.allclose(tt_matvec(cores, x / scales), y_unscaled, atol=1e-4, rtol=1e-4)
