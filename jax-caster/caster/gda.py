"""The Gaussian discriminant of eq. (5).

    g_c(z) = -0.5 (z - mu_c)^T Sigma_c^-1 (z - mu_c) - 0.5 log det Sigma_c + log pi_c

Prediction is argmax_c g_c(z). Feeding it the transported moments of eq. (4)
gives CASTER's adapted classifier; feeding it the stored source moments gives
the untransported Source-GDA baseline of Table 1, which is also what the gate
falls back to when no frozen head is supplied.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .linalg import psd_logdet_and_quadform, ridge, symmetrize


def discriminant_scores(z, means, covs, priors):
    """Equation (5) for a batch, returning (B, C) scores."""
    z = jnp.asarray(z)
    means = jnp.asarray(means)
    covs = symmetrize(jnp.asarray(covs))
    residuals = z[:, None, :] - means[None, :, :]          # (B, C, k)
    logdet, quad = psd_logdet_and_quadform(covs[None], residuals)
    log_priors = jnp.log(jnp.clip(jnp.asarray(priors), jnp.finfo(z.dtype).tiny, None))
    return -0.5 * quad - 0.5 * logdet + log_priors[None, :]


def source_scores(state, z, beta):
    """Equation (5) with the stored source moments, no transport."""
    return discriminant_scores(z, state.means, ridge(state.covs, beta), state.priors)


def predict(scores):
    """argmax_c of eq. (5)."""
    return jnp.argmax(scores, axis=-1)


def confidence(scores):
    """max_c softmax(g)_c, the retention statistic of eq. (6)."""
    return jnp.max(jax.nn.softmax(scores, axis=-1), axis=-1)
