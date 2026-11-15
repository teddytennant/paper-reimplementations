"""NVFP4 group quantizer: E2M1 codes, E4M3 scales, and Q_s."""

import jax.numpy as jnp

# E2M1 levels. Round-to-nearest, ties closer to zero (section 4.1.1).
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_BIAS = 7
# Positive finite E4M3 codes. exp=15 and mantissa=7 is NaN and is excluded.
E4M3_CODE_COUNT = 126


def e4m3_positive_codes(dtype=None):
    """Positive finite E4M3 values, ascending. Zero and the NaN code are excluded.

    E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7.
    Subnormals use exponent 0. The all-ones exponent with mantissa 7 is NaN.
    The largest finite code is 448.
    """
    values = []
    for exp in range(16):
        for mant in range(8):
            if exp == 0 and mant == 0:
                continue
            if exp == 15 and mant == 7:
                continue
            if exp == 0:
                value = mant * 2.0 ** (-9)
            else:
                value = (1.0 + mant / 8.0) * 2.0 ** (exp - E4M3_BIAS)
            values.append(value)
    codes = jnp.asarray(values, dtype=dtype) if dtype is not None else jnp.asarray(values)
    return codes


def gamma_from_amax(amax):
    """Tensor scale gamma = (448 * 6) / amax (equation 27)."""
    amax = jnp.asarray(amax)
    return (E4M3_MAX * E2M1_MAX) / amax


def exportable_scales(amax, dtype=None):
    """S = {a / gamma : a in E4M3+}, ascending (equations 6 and 27)."""
    amax = jnp.asarray(amax)
    out_dtype = dtype if dtype is not None else amax.dtype
    codes = e4m3_positive_codes(dtype=out_dtype)
    gamma = jnp.asarray(gamma_from_amax(amax), dtype=out_dtype)
    return codes / gamma


def rho(x):
    """Round-to-nearest onto E2M1 levels. Ties take the level closer to zero.

    Midpoints are assigned by searchsorted(..., side='left') on the midpoints,
    which selects the lower magnitude. Values beyond 6 saturate at 6.
    """
    x = jnp.asarray(x)
    levels = jnp.asarray(E2M1_LEVELS, dtype=x.dtype)
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    magnitude = jnp.abs(x)
    index = jnp.searchsorted(midpoints, magnitude, side="left")
    rounded = levels[index]
    return jnp.sign(x) * rounded


def quantize(w, s):
    """Q_s(w) = s * rho(w / s) (equation 5).

    s broadcasts against w. A per-row scale of shape (M,) is expanded over a
    group of shape (M, B).
    """
    w = jnp.asarray(w)
    s = jnp.asarray(s, dtype=w.dtype)
    if s.ndim == 1 and w.ndim >= 2 and s.shape[0] == w.shape[0]:
        s = s.reshape(s.shape + (1,) * (w.ndim - 1))
    return s * rho(w / s)
