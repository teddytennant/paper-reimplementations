"""The spectral machinery has to be exact, and the degenerate case has to reduce.

Checks two things about ``flf.losses``:

* Parseval. With the unnormalized DFT, sum_{u,v} |F(e)|^2 = H * W * sum |e|^2.
  If this does not hold, every statement about "the error energy at frequency k"
  in the paper's analysis is measuring something else.
* The reduction of Eq. (3). Dropping the focal weight (focal_exponent = 0) and
  replacing log(1 + e_uv) with e_uv^2 turns f-loss into a plain spectral L2,
  which by Parseval is exactly H * W times the pixel v-loss of Eq. (2),
  including the shared 1/(1-t)^2 timestep weight.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from flf.losses import f_loss, spectral_residual, timestep_weight, v_loss
from flf.spectral import fft2

# float64 FFTs agree with the direct sum to well past this; 1e-11 leaves room
# for cuFFT differing from the CPU implementation in the last few bits.
RTOL = 1e-11


def _pair(seed, shape=(4, 16, 16, 3)):
    ka, kb = jax.random.split(jax.random.key(seed))
    return (
        jax.random.normal(ka, shape, dtype=jnp.float64),
        jax.random.normal(kb, shape, dtype=jnp.float64),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parseval_holds_for_the_residual_spectrum(seed):
    pred, target = _pair(seed)
    err = pred - target
    height, width = err.shape[-3], err.shape[-2]
    spectral = jnp.sum(jnp.abs(fft2(err)) ** 2)
    pixel = jnp.sum(err**2)
    assert np.isclose(spectral, height * width * pixel, rtol=RTOL, atol=0.0)


def test_spectral_residual_matches_a_direct_abs():
    pred, target = _pair(7, shape=(2, 8, 8, 1))
    e, e_sq = spectral_residual(pred, target)
    direct = jnp.abs(fft2(pred - target))
    assert np.allclose(e, direct, rtol=RTOL, atol=0.0)
    assert np.allclose(e_sq, direct**2, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_focal_exponent_zero_with_square_term_is_pixel_l2(seed):
    pred, target = _pair(seed)
    t = jnp.linspace(0.05, 0.95, pred.shape[0], dtype=jnp.float64)
    height, width = pred.shape[-3], pred.shape[-2]
    degenerate = f_loss(pred, target, t, focal_exponent=0.0, spectral_term="square")
    assert np.isclose(degenerate, height * width * v_loss(pred, target, t), rtol=RTOL)


def test_reduction_survives_the_clipped_timestep_weight():
    """The 1/(1-t)^2 factor of Table 5 is shared, so it must cancel at any t."""
    pred, target = _pair(3, shape=(6, 8, 8, 1))
    t = jnp.array([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=jnp.float64)
    assert float(jnp.max(timestep_weight(t))) == pytest.approx(1.0 / 0.05**2)
    degenerate = f_loss(pred, target, t, focal_exponent=0.0, spectral_term="square")
    assert np.isclose(degenerate, 64 * v_loss(pred, target, t), rtol=RTOL)


def test_zero_residual_gives_zero_loss_and_finite_gradient():
    """log(1 + e) at e = 0 must not produce a NaN gradient."""
    x = jax.random.normal(jax.random.key(11), (2, 8, 8, 1), dtype=jnp.float64)
    t = jnp.full((2,), 0.3, dtype=jnp.float64)
    assert float(f_loss(x, x, t)) == 0.0
    grad = jax.grad(lambda p: f_loss(p, x, t))(x)
    assert np.all(np.isfinite(np.asarray(grad)))
