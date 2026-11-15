"""Check the precision the exactness tests depend on, rather than assuming it."""

import jax.numpy as jnp
import numpy as np

from pref_mh.x64 import is_x64


def test_x64_took_effect():
    assert is_x64()
    assert jnp.zeros(3).dtype == jnp.dtype("float64")
    assert jnp.asarray(1.0).dtype == jnp.dtype("float64")


def test_matmul_is_not_tf32():
    """A matmul whose answer TF32 would get wrong in the 4th digit.

    The stationarity test is a matmul, so this is the failure mode that would
    make it meaningless: TF32 keeps 10 mantissa bits, so the 1e-4 perturbation
    below would come back with roughly 1e-3 relative error.
    """
    rng = np.random.default_rng(0)
    a = 1.0 + 1e-4 * rng.standard_normal((64, 64))
    b = 1.0 + 1e-4 * rng.standard_normal((64, 64))
    got = np.asarray(jnp.asarray(a) @ jnp.asarray(b))
    want = a @ b
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-13
