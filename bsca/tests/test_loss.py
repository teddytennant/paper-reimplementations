"""N3 and temporal regularizers."""

import jax.numpy as jnp
import numpy as np

from bsca import n3_penalty, temporal_penalty


def test_n3_nonnegative_zero_only_at_zeros():
    z = jnp.zeros((4, 3), dtype=jnp.complex64)
    np.testing.assert_allclose(float(n3_penalty(z)), 0.0, atol=1e-8)
    nz = z.at[1, 2].set(1.0 + 0.0j)
    val = float(n3_penalty(nz))
    assert val >= 0.0
    assert val > 0.0
    both = float(n3_penalty(z, nz))
    np.testing.assert_allclose(both, val, atol=1e-6)
    neg = z.at[0, 0].set(-2.0 + 0.0j)
    np.testing.assert_allclose(float(n3_penalty(neg)), 8.0, atol=1e-5)


def test_temporal_penalty_zero_iff_constant():
    equal = jnp.ones((5, 4, 2), dtype=jnp.complex64) * (0.3 + 0.1j)
    np.testing.assert_allclose(float(temporal_penalty(equal, p=3)), 0.0, atol=1e-7)
    differ = equal.at[2, 0, 0].set(1.0 + 0.0j)
    assert float(temporal_penalty(differ, p=3)) > 0.0
    params = {
        "time_mul": equal,
        "time_add": equal,
        "time_joint": equal,
    }
    np.testing.assert_allclose(float(temporal_penalty(params, p=3)), 0.0, atol=1e-7)
    params2 = dict(params)
    params2["time_mul"] = differ
    assert float(temporal_penalty(params2, p=3)) > 0.0
