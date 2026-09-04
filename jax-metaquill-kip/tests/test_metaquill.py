"""MetaQuill: Eqs. 13 and 15, closed-form adaptation, optional meta-train."""

import jax
import jax.numpy as jnp
import numpy as np

from metaquill_kip.data import coord_grid, gaussian_blob, stroke_tasks
from metaquill_kip.field import MLPConfig, encode, encoded_dim, init_params, make_field
from metaquill_kip.metaquill import (
    flat_field,
    kernel_coefficients,
    lift_to_weights,
    meta_train,
    metaquill_adapt,
    tangent_predict,
)
from metaquill_kip.metrics import psnr
from metaquill_kip.ntk import jacobian_matrix, kernel_predict, ntk_matrix


K_BANDS = 3


def _field(seed=0, width=16, depth=2):
    cfg = MLPConfig(in_dim=encoded_dim(2, "fourier", K_BANDS), width=width, depth=depth)
    params = init_params(cfg, jax.random.PRNGKey(seed))
    return make_field(cfg), params, lambda x: encode(x, "fourier", K_BANDS)


def test_lift_to_weights_matches_kernel_predict():
    """Eqs. 13 and 15: J delta = K w, so tangent_predict equals kernel_predict."""
    f, params, enc = _field()
    coords = coord_grid(8)
    Z = enc(jnp.asarray(coords))
    Y = jnp.asarray(gaussian_blob(8))
    g, theta, _ = flat_field(f, params)
    w = kernel_coefficients(f, params, Z, Y, ridge=1e-6)
    delta = lift_to_weights(g, theta, Z, w)
    K = ntk_matrix(f, params, Z)
    tan = tangent_predict(g, theta, delta, Z)
    ker = kernel_predict(K, w)
    np.testing.assert_allclose(np.asarray(tan), np.asarray(ker), atol=1e-8, rtol=1e-8)


def test_directional_derivative_matches_j_delta():
    f, params, enc = _field()
    Z = enc(jnp.asarray(coord_grid(6)))
    g, theta, _ = flat_field(f, params)
    w = jax.random.normal(jax.random.PRNGKey(3), (Z.shape[0],), dtype=jnp.float64)
    delta = lift_to_weights(g, theta, Z, w)
    jvp_out = tangent_predict(g, theta, delta, Z)
    J = jacobian_matrix(f, params, Z)
    np.testing.assert_allclose(np.asarray(jvp_out), np.asarray(J @ delta), atol=1e-8)

    eps = 1e-6
    fd = (g(theta + eps * delta, Z) - g(theta - eps * delta, Z)) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(fd), np.asarray(jvp_out), atol=1e-6, rtol=1e-5)


def test_metaquill_adapt_psnr_on_full_blob():
    f, params, enc = _field()
    coords = coord_grid(8)
    Z = enc(jnp.asarray(coords))
    Y = jnp.asarray(gaussian_blob(8))
    delta, w = metaquill_adapt(f, params, Z, Y, ridge=1e-6)
    g, theta, _ = flat_field(f, params)
    pred = tangent_predict(g, theta, delta, Z)
    db = float(psnr(pred, Y))
    assert db > 20.0


def test_meta_train_loss_decreases():
    f, params, enc = _field(seed=1)
    Z = enc(jnp.asarray(coord_grid(8)))
    Y_tasks = stroke_tasks(8, names=("seven", "one", "l"))
    _, _, hist = meta_train(f, params, Z, Y_tasks, steps=40, lr=3e-3, l2=1e-3)
    assert float(hist[-1]) < float(hist[0])
    assert jnp.isfinite(hist[-1])
