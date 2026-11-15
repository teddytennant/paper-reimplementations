"""MetaQuill-KIP hybrid of Sec. III-C at toy scale."""

import jax
import jax.numpy as jnp
import numpy as np

from metaquill_kip.data import center_hole_mask, coord_grid, gaussian_blob
from metaquill_kip.field import MLPConfig, encode, encoded_dim, init_params, make_field
from metaquill_kip.kip import init_support, kip_predict, ntk_kip
from metaquill_kip.metaquill import flat_field, tangent_predict
from metaquill_kip.metaquill_kip import kip_warm_start, refine_delta
from metaquill_kip.metrics import psnr


SEED = 0
K_BANDS = 3
RIDGE = 1e-4


def _setup(seed=SEED):
    cfg = MLPConfig(in_dim=encoded_dim(2, "fourier", K_BANDS), width=16, depth=2)
    params = init_params(cfg, jax.random.PRNGKey(seed))
    f = make_field(cfg)
    encode_fn = lambda x: encode(x, "fourier", K_BANDS)
    coords = coord_grid(8)
    target = gaussian_blob(8)
    return f, params, encode_fn, coords, target


def test_kip_warm_start_matches_kip_predict():
    """Eq. 10: tangent_predict(delta_KIP) equals kip_predict at query points."""
    f, params, encode_fn, coords, target = _setup()
    obs = center_hole_mask(8, 8, radius=0.45)
    support, labels = init_support(coords[obs], target[obs], m=6, seed=SEED + 2, space="coord", encode_fn=encode_fn)
    Z_obs = encode_fn(jnp.asarray(coords[obs]))
    Y_obs = jnp.asarray(target[obs])
    support, labels, _ = ntk_kip(
        f, params, Z_obs, Y_obs, support, labels, ridge=RIDGE, encode_fn=encode_fn, space="coord", steps=8, lr=1e-2
    )
    Z_query = encode_fn(jnp.asarray(coords))
    kip_pred, _ = kip_predict(f, params, support, labels, Z_query, RIDGE, encode_fn, space="coord")
    delta, _ = kip_warm_start(f, params, support, labels, RIDGE, encode_fn, space="coord")
    g, theta, _ = flat_field(f, params)
    tan = tangent_predict(g, theta, delta, Z_query)
    np.testing.assert_allclose(np.asarray(tan), np.asarray(kip_pred), atol=1e-8, rtol=1e-8)


def test_refine_delta_does_not_explode_vs_kip_warm_start():
    """On a center-hole 8x8 blob, a few refine steps stay within 1 dB of KIP (or better)."""
    f, params, encode_fn, coords, target = _setup(seed=SEED)
    obs = center_hole_mask(8, 8, radius=0.45)
    support, labels = init_support(coords[obs], target[obs], m=6, seed=SEED + 2, space="coord", encode_fn=encode_fn)
    Z_obs = encode_fn(jnp.asarray(coords[obs]))
    Y_obs = jnp.asarray(target[obs])
    support, labels, _ = ntk_kip(
        f, params, Z_obs, Y_obs, support, labels, ridge=RIDGE, encode_fn=encode_fn, space="coord", steps=20, lr=2e-2
    )
    delta_kip, _ = kip_warm_start(f, params, support, labels, RIDGE, encode_fn, space="coord")
    g, theta, _ = flat_field(f, params)
    kip_obs = tangent_predict(g, theta, delta_kip, Z_obs)
    kip_db = float(psnr(kip_obs, Y_obs))
    delta, hist = refine_delta(
        f, params, delta_kip, Z_obs, Y_obs, steps=15, lr=1e-3, l2=1e-6, centered=True
    )
    assert jnp.all(jnp.isfinite(hist))
    assert jnp.isfinite(delta).all()
    refined_obs = tangent_predict(g, theta, delta, Z_obs)
    refined_db = float(psnr(refined_obs, Y_obs))
    assert refined_db + 1.0 >= kip_db
