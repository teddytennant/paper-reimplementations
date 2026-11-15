"""Activation importance and diagonal column scales.

Std uses the population variance (ddof=0): divide by T, not T-1. That is the
Var_t the method writes as a population second central moment. eps is added
inside the square root, not after it.

MeanAbs does not use eps. The argument is kept so both statistics share the
(C, eps) signature used by compress.
"""

import jax.numpy as jnp

# Convenience defaults. The paper ablates alpha in {0.5, 1.0, 1.5} and does
# not publish a single default for eps or min_scale.
DEFAULT_ALPHA = 1.0
DEFAULT_EPS = 1e-6
DEFAULT_MIN_SCALE = 1e-3


def importance_mean_abs(C, eps=DEFAULT_EPS):
    """Mean absolute activation per input channel.

    imp_i = mean_t |C[t, i]|. eps is accepted and ignored.
    """
    del eps
    C = jnp.asarray(C)
    return jnp.mean(jnp.abs(C), axis=0)


def importance_std(C, eps=DEFAULT_EPS):
    """Population standard deviation per input channel, with eps inside the sqrt.

    imp_i = sqrt(Var_t(C[:, i]) + eps), Var_t = population variance (ddof=0).
    """
    C = jnp.asarray(C)
    variance = jnp.var(C, axis=0, ddof=0)
    return jnp.sqrt(variance + eps)


def column_scales(imp, alpha=DEFAULT_ALPHA, eps=DEFAULT_EPS, min_scale=DEFAULT_MIN_SCALE):
    """Positive column scales from per-channel importance.

    imp_bar = mean_i imp_i
    imp_eps = max(imp_bar, eps)
    s_min = max(min_scale, eps)
    s_i = max(s_min, (imp_i / imp_eps) ** alpha)
    """
    imp = jnp.asarray(imp)
    imp_bar = jnp.mean(imp)
    imp_eps = jnp.maximum(imp_bar, eps)
    s_min = jnp.maximum(min_scale, eps)
    scaled = (imp / imp_eps) ** alpha
    return jnp.maximum(s_min, scaled)
