"""The central claim: v-loss gradients are low-frequency dominated, f-loss is not.

From the abstract and Section 3.2 of arXiv:2609.02748: natural images carry a
1/f^2 power spectrum, pixel objectives treat all spatial errors uniformly, so
the low frequencies dominate the optimization signal and fine detail arrives
late. f-loss is supposed to equalize the learning signal across frequencies.

Measured here on the data rather than asserted from the abstract. Take a batch
with a controlled 1/f^2 spectrum, take the gradient of each loss with respect to
the prediction, transform it, and average |grad|^2 per DFT bin inside six radial
rings. Per bin, not per ring, so ring area is divided out and a flat profile
means every mode pulls equally hard. ``spread`` is the ratio of the loudest ring
to the quietest.

Two error regimes, both with the 1/f^2 shape the paper's argument is about:
the prediction at initialization (error = the image itself), and a prediction
already close to the target but off by an independent 1/f^2 draw.

Measured values, four seeds, both regimes: v-loss spread 20.1 to 23.1, f-loss
spread 1.25 to 1.47, ratio 13.7 to 17.5. The thresholds below sit well inside
that.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from flf.data import sample_images
from flf.losses import f_loss, v_loss
from flf.spectral import band_means, fft2

N_BANDS = 6
BATCH = 64


def _band_profile(loss_fn, pred):
    """Mean |dL/dpred|^2 per DFT bin in each radial ring, peak-normalized."""
    grad = jax.grad(loss_fn)(pred)
    bands = band_means(jnp.abs(fft2(grad)) ** 2, N_BANDS)
    return np.asarray(bands / jnp.max(bands))


def _spread(profile):
    return float(profile.max() / profile.min())


def _setup(seed, regime):
    k_data, k_err = jax.random.split(jax.random.key(seed))
    target = sample_images(k_data, BATCH, dtype=jnp.float64)
    t = jnp.full((BATCH,), 0.5, dtype=jnp.float64)
    if regime == "at_init":
        pred = jnp.zeros_like(target)
    else:
        pred = target + 0.5 * sample_images(k_err, BATCH, dtype=jnp.float64)
    return pred, target, t


def test_synthetic_data_actually_has_a_1_over_f_squared_spectrum():
    """No point testing the losses if the data is not the paper's setting."""
    x = sample_images(jax.random.key(0), 512, dtype=jnp.float64)
    power = np.asarray(band_means(jnp.abs(fft2(x)) ** 2, N_BANDS))
    assert power[0] / power[-1] > 15.0
    assert np.all(np.diff(power[:4]) < 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("regime", ["at_init", "residual_error"])
def test_f_loss_gradient_is_far_flatter_across_frequency(seed, regime):
    pred, target, t = _setup(seed, regime)
    v_profile = _band_profile(lambda p: v_loss(p, target, t), pred)
    f_profile = _band_profile(lambda p: f_loss(p, target, t), pred)

    # v-loss: loudest ring is the lowest one, and it drowns out the rest.
    assert int(np.argmax(v_profile)) == 0
    assert _spread(v_profile) > 12.0

    # f-loss: no ring is twice as loud as any other.
    assert _spread(f_profile) < 2.0
    assert _spread(v_profile) / _spread(f_profile) > 8.0


@pytest.mark.parametrize("seed", [0, 1])
def test_both_the_focal_weight_and_the_log_are_needed(seed):
    """Table 6: dropping either component costs most of the balancing.

    Row "w_uv = 1" is focal_exponent 0, and the prior focal frequency loss the
    paper compares against is the squared term with the focal weight kept.
    Neither one flattens the gradient the way Eq. (3) does.
    """
    pred, target, t = _setup(seed, "at_init")
    full = _spread(_band_profile(lambda p: f_loss(p, target, t), pred))
    no_focal = _spread(
        _band_profile(lambda p: f_loss(p, target, t, focal_exponent=0.0), pred)
    )
    no_log = _spread(
        _band_profile(lambda p: f_loss(p, target, t, spectral_term="square"), pred)
    )
    assert full < 2.0
    assert no_focal > 5.0
    assert no_log > 30.0


def test_log_compression_is_what_shrinks_large_residuals():
    """Per-frequency gradient magnitude is w_uv / (1 + e_uv), not w_uv * e_uv.

    That 1/(1 + e) is the whole mechanism: it flattens exactly where the 1/f^2
    residuals are large, which is the low frequencies.
    """
    pred, target, t = _setup(0, "at_init")
    e = jnp.abs(fft2(pred - target))
    low = jnp.mean(e[:, :2, :2, :])
    high = jnp.mean(e[:, 7:9, 7:9, :])
    # Measured: the mean residual is 6.5x larger in the lowest DFT bins than
    # near Nyquist, and the log's 1/(1 + e) factor hands back a 5.7x advantage
    # to the high bins, which is what cancels the imbalance.
    assert float(low) > 5.0 * float(high)
    assert float(1.0 / (1.0 + high)) > 4.0 * float(1.0 / (1.0 + low))
