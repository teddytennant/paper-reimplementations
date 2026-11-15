"""NTK-KIP (Eq. 9): inducing points through the finite NTK."""

import jax
import jax.numpy as jnp
import numpy as np

from metaquill_kip.data import coord_grid, gaussian_blob, random_mask
from metaquill_kip.field import MLPConfig, encode, encoded_dim, init_params, make_field
from metaquill_kip.kip import init_support, kip_loss, kip_predict, ntk_kip


SEED = 0
K_BANDS = 3


def _blob_problem(h=8, fraction=0.4, seed=SEED, m=6):
    cfg = MLPConfig(in_dim=encoded_dim(2, "fourier", K_BANDS), width=16, depth=2)
    params = init_params(cfg, jax.random.PRNGKey(seed))
    f = make_field(cfg)
    encode_fn = lambda x: encode(x, "fourier", K_BANDS)
    coords = coord_grid(h)
    target = gaussian_blob(h)
    obs = random_mask(target.size, fraction, seed + 1)
    Z_obs = encode_fn(jnp.asarray(coords[obs]))
    Y_obs = jnp.asarray(target[obs])
    support, labels = init_support(coords[obs], target[obs], m=m, seed=seed + 2, space="coord", encode_fn=encode_fn)
    return f, params, encode_fn, coords, target, obs, Z_obs, Y_obs, support, labels


def test_kip_loss_is_finite():
    f, params, encode_fn, _, _, _, Z_obs, Y_obs, support, labels = _blob_problem()
    loss = kip_loss(f, params, support, labels, Z_obs, Y_obs, ridge=1e-4, encode_fn=encode_fn, space="coord")
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0


def test_ntk_kip_drops_observed_mse():
    """After 40 steps on an 8x8 blob with m=6, observed MSE drops vs init (Eq. 9)."""
    f, params, encode_fn, _, _, _, Z_obs, Y_obs, support, labels = _blob_problem(seed=SEED, m=6)
    ridge = 1e-4
    mse0 = float(kip_loss(f, params, support, labels, Z_obs, Y_obs, ridge, encode_fn, "coord"))
    support_f, labels_f, hist = ntk_kip(
        f,
        params,
        Z_obs,
        Y_obs,
        support,
        labels,
        ridge=ridge,
        encode_fn=encode_fn,
        space="coord",
        steps=40,
        lr=2e-2,
    )
    mse1 = float(kip_loss(f, params, support_f, labels_f, Z_obs, Y_obs, ridge, encode_fn, "coord"))
    assert mse1 < mse0
    assert float(hist[-1]) < float(hist[0])


def test_kip_predict_interpolates_support():
    f, params, encode_fn, _, _, _, Z_obs, Y_obs, support, labels = _blob_problem(seed=SEED, m=6)
    support_f, labels_f, _ = ntk_kip(
        f,
        params,
        Z_obs,
        Y_obs,
        support,
        labels,
        ridge=1e-6,
        encode_fn=encode_fn,
        space="coord",
        steps=8,
        lr=1e-2,
    )
    Zs = encode_fn(support_f)
    pred, _ = kip_predict(f, params, support_f, labels_f, Zs, 1e-6, encode_fn, space="coord")
    np.testing.assert_allclose(np.asarray(pred), np.asarray(labels_f), atol=1e-4, rtol=1e-4)
