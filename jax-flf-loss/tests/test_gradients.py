"""Finite differences against the analytic gradient of f-loss.

The focal weight of Eq. (3) is taken under a stop-gradient, so f-loss as
implemented is not the derivative of the expression you get by reading Eq. (3)
as a plain function of the prediction. Both halves of that are checked:

* focal_exponent = 0 removes the weight entirely, leaving sum log(1 + e_uv),
  a smooth function whose gradient can be differenced directly.
* for the paper's focal_exponent = 1, the analytic gradient is compared against
  central differences of the surrogate that freezes the weight at the base
  point, which is exactly what stop-gradient promises.

All of this runs in float64. In default float32 the differencing noise alone is
around 1e-2 relative.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from flf.losses import f_loss, spectral_residual
from flf.spectral import fft2

STEP = 1e-5


def _central_differences(fn, x):
    base = np.asarray(x)
    out = np.zeros_like(base)
    flat = base.reshape(-1)
    for i in range(flat.size):
        up = flat.copy()
        up[i] += STEP
        down = flat.copy()
        down[i] -= STEP
        plus = fn(jnp.asarray(up.reshape(base.shape)))
        minus = fn(jnp.asarray(down.reshape(base.shape)))
        out.reshape(-1)[i] = (plus - minus) / (2 * STEP)
    return out


@pytest.fixture
def small_problem():
    ka, kb = jax.random.split(jax.random.key(5))
    shape = (2, 4, 4, 1)
    pred = jax.random.normal(ka, shape, dtype=jnp.float64)
    target = jax.random.normal(kb, shape, dtype=jnp.float64)
    t = jnp.array([0.2, 0.7], dtype=jnp.float64)
    return pred, target, t


def test_unweighted_log_gradient_matches_finite_differences(small_problem):
    pred, target, t = small_problem

    def loss(p):
        return f_loss(p, target, t, focal_exponent=0.0)

    numeric = _central_differences(loss, pred)
    analytic = np.asarray(jax.grad(loss)(pred))
    assert np.allclose(analytic, numeric, rtol=2e-6, atol=1e-8)


def test_focal_gradient_matches_the_frozen_weight_surrogate(small_problem):
    pred, target, t = small_problem
    e0, _ = spectral_residual(pred, target)
    weight = e0 / jnp.max(e0, axis=(-3, -2, -1), keepdims=True)
    scale = 1.0 / jnp.maximum(1.0 - t, 0.05) ** 2

    def surrogate(p):
        e = jnp.abs(fft2(p - target))
        return jnp.mean(scale * jnp.sum(weight * jnp.log1p(e), axis=(-3, -2, -1)))

    # Same value, so the surrogate really is Eq. (3) at this point.
    assert np.isclose(surrogate(pred), f_loss(pred, target, t), rtol=1e-12)

    numeric = _central_differences(surrogate, pred)
    analytic = np.asarray(jax.grad(lambda p: f_loss(p, target, t))(pred))
    assert np.allclose(analytic, numeric, rtol=2e-6, atol=1e-8)


def test_focal_gradient_ignores_the_weight_path():
    """Without the stop-gradient the weight would contribute; check it does not."""
    ka, kb = jax.random.split(jax.random.key(21))
    pred = jax.random.normal(ka, (2, 4, 4, 1), dtype=jnp.float64)
    target = jax.random.normal(kb, (2, 4, 4, 1), dtype=jnp.float64)
    t = jnp.full((2,), 0.4, dtype=jnp.float64)

    def unfrozen(p):
        e = jnp.abs(fft2(p - target))
        w = e / jnp.max(e, axis=(-3, -2, -1), keepdims=True)
        scale = 1.0 / jnp.maximum(1.0 - t, 0.05) ** 2
        return jnp.mean(scale * jnp.sum(w * jnp.log1p(e), axis=(-3, -2, -1)))

    frozen = np.asarray(jax.grad(lambda p: f_loss(p, target, t))(pred))
    loose = np.asarray(jax.grad(unfrozen)(pred))
    assert not np.allclose(frozen, loose, rtol=1e-3)
