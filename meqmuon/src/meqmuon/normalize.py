"""RMS normalization and coefficient of variation."""

import jax.numpy as jnp


def row_rms(x):
    """Per-row RMS, r_i = sqrt(mean_j X_ij^2)."""
    x = jnp.asarray(x)
    return jnp.sqrt(jnp.mean(x * x, axis=1))


def column_rms(x):
    """Per-column RMS, c_j = sqrt(mean_i X_ij^2)."""
    x = jnp.asarray(x)
    return jnp.sqrt(jnp.mean(x * x, axis=0))


def coefficient_of_variation(rms, delta=1e-7):
    """Population CV over entries with RMS > delta. Empty or zero mean returns 0."""
    rms = jnp.asarray(rms)
    delta = jnp.asarray(delta, dtype=rms.dtype)
    mask = rms > delta
    count = jnp.sum(mask)
    count_f = jnp.maximum(count, jnp.asarray(1, dtype=count.dtype)).astype(rms.dtype)
    total = jnp.sum(jnp.where(mask, rms, jnp.zeros_like(rms)))
    mean = total / count_f
    centered = jnp.where(mask, rms - mean, jnp.zeros_like(rms))
    second = jnp.sum(centered * centered) / count_f
    gamma = jnp.sqrt(second) / mean
    invalid = (count == 0) | (mean == 0)
    return jnp.where(invalid, jnp.asarray(0.0, dtype=rms.dtype), gamma)


def row_normalize(x, delta=1e-7):
    """Rescale rows with RMS > delta to unit RMS. Smaller rows are left unchanged."""
    x = jnp.asarray(x)
    rms = row_rms(x)
    scale = jnp.where(rms > jnp.asarray(delta, dtype=x.dtype), rms, jnp.ones_like(rms))
    return x / scale[:, None]


def column_normalize(x, delta=1e-7):
    """Rescale columns with RMS > delta to unit RMS. Smaller columns are unchanged."""
    x = jnp.asarray(x)
    rms = column_rms(x)
    scale = jnp.where(rms > jnp.asarray(delta, dtype=x.dtype), rms, jnp.ones_like(rms))
    return x / scale[None, :]


def global_rms_normalize(x):
    """Scale a vector so its RMS is 1. An exact-zero vector returns zeros."""
    x = jnp.asarray(x)
    nrm = jnp.linalg.norm(x)
    length = jnp.asarray(x.shape[0], dtype=x.dtype)
    safe = jnp.where(nrm == 0, jnp.ones_like(nrm), nrm)
    inv = jnp.where(nrm == 0, jnp.zeros_like(nrm), jnp.sqrt(length) / safe)
    return x * inv


def one_sided_normalize(u, delta=1e-7):
    """Row-normalize when gamma_row >= gamma_col, otherwise column-normalize."""
    u = jnp.asarray(u)
    gamma_row = coefficient_of_variation(row_rms(u), delta)
    gamma_col = coefficient_of_variation(column_rms(u), delta)
    row_n = row_normalize(u, delta)
    col_n = column_normalize(u, delta)
    return jnp.where(gamma_row >= gamma_col, row_n, col_n)
