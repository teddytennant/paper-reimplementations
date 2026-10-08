"""Hand checks for symmetric quantization."""

import jax.numpy as jnp
import pytest

from lea_ptq.quant import symmetric_quantize


def test_b3_vector_uses_jax_round():
    # bits=3, qmax=3, max abs=1, scale=1/3.
    # W/scale = [3, -1.5, 0.75].
    # jnp.round(3) = 3, jnp.round(-1.5) = -2, jnp.round(0.75) = 1.
    # Dequantized outputs are [1, -2/3, 1/3].
    rounded_half = jnp.round(jnp.asarray(-1.5))
    assert float(rounded_half) == -2.0

    quantized = symmetric_quantize(jnp.asarray([1.0, -0.5, 0.25]), 3)
    assert quantized.shape == (3,)
    assert float(quantized[0]) == pytest.approx(1.0)
    assert float(quantized[1]) == pytest.approx(-2.0 / 3.0)
    assert float(quantized[2]) == pytest.approx(1.0 / 3.0)


def test_rows_have_independent_scales_and_zero_row_stays_zero():
    weights = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
        ]
    )
    quantized = symmetric_quantize(weights, 3)
    # Row scales are 1/3, 1.0 (zero row), and 0.1/3. A shared scale of 1/3
    # would round 0.1/(1/3)=0.3 to 0 and wipe the third row.
    assert float(quantized[0, 0]) == pytest.approx(1.0)
    assert float(quantized[0, 1]) == pytest.approx(0.0)
    assert float(quantized[1, 0]) == pytest.approx(0.0)
    assert float(quantized[1, 1]) == pytest.approx(0.0)
    assert float(quantized[2, 0]) == pytest.approx(0.1)
    assert float(quantized[2, 1]) == pytest.approx(0.0)
    assert float(jnp.max(jnp.abs(weights[0]))) / 3.0 != pytest.approx(
        float(jnp.max(jnp.abs(weights[2]))) / 3.0
    )


def test_bits_below_two_raises():
    with pytest.raises(ValueError):
        symmetric_quantize(jnp.asarray([1.0, 0.5]), 1)
